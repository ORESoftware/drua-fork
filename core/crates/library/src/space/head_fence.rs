use std::path::Path;
use std::sync::Arc;

use obix::out::{EphemeralEventType, Outbox};
use obix::MailboxConfig;
use serde::{Deserialize, Serialize};
use sqlx::types::Json;
use tokio::sync::OnceCell;

use crate::git::GitEngine;

use super::SpaceError;

const LIBRARY_HEAD_EVENT_TYPE: EphemeralEventType =
    EphemeralEventType::new("drua_library_head");

/// Separate from the Git writer's push lock. This lock serializes the
/// post-push remote-head snapshot + durable fence publication across replicas.
/// `0x64727561666e63` = "druafnc".
const LIBRARY_HEAD_FENCE_LOCK_KEY: i64 = 0x64727561666e63;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LibraryHeadFence {
    head: String,
}

/// Cross-replica read-your-writes barrier for space reads.
///
/// A successful write is not considered complete at the `Spaces` API boundary
/// until the pushed Git head is persisted as an obix ephemeral event. Reads
/// consult that durable fence before touching the local bare clone. If the
/// local clone does not contain the required head, the read synchronously
/// fetches origin before serving data.
///
/// Notifications remain an acceleration mechanism; the persisted obix row is
/// the correctness boundary. This matters because a wake-up can arrive before
/// another replica has actually advanced its local HEAD.
#[derive(Clone)]
pub(super) struct HeadFence {
    git: Arc<GitEngine>,
    pool: sqlx::PgPool,
    outbox: Arc<OnceCell<Outbox<LibraryHeadFence>>>,
}

impl HeadFence {
    pub(super) fn new(git: &Arc<GitEngine>, pool: &sqlx::PgPool) -> Self {
        return Self {
            git: Arc::clone(git),
            pool: pool.clone(),
            outbox: Arc::new(OnceCell::new()),
        };
    }

    async fn outbox(&self) -> Result<&Outbox<LibraryHeadFence>, SpaceError> {
        return self
            .outbox
            .get_or_try_init(|| async {
                let config = MailboxConfig::builder()
                    .build()
                    .map_err(|e| SpaceError::Git(format!("obix mailbox config: {e}")))?;
                let outbox = Outbox::<LibraryHeadFence>::init(&self.pool, config)
                    .await
                    .map_err(|e| SpaceError::Git(format!("obix outbox init: {e}")))?;
                return Ok(outbox);
            })
            .await;
    }

    /// Publish the newest pushed remote head as the durable visibility fence.
    ///
    /// Fence publication is serialized cluster-wide with a transaction-scoped
    /// advisory lock. The same transaction persists the obix event, so the lock
    /// holder never needs a second pool connection while waiters are blocked.
    /// Transaction scope also makes lock release cancellation-safe: dropping the
    /// future rolls the transaction back and Postgres releases the lock.
    ///
    /// The fence lock is deliberately distinct from the Git push lock: taking
    /// the push lock here would invert the writer's `repo_mutex -> push_lock`
    /// order and could deadlock with the next local batch. Under the fence lock
    /// we fetch origin, so a late publisher can only publish the same or a newer
    /// remote head than an earlier publisher.
    pub(super) async fn after_write(&self) -> Result<(), SpaceError> {
        // Initialize before opening the fence transaction. Obix initialization
        // may need pool connections of its own and must never run while holding
        // the advisory lock.
        let outbox = self.outbox().await?;

        let mut tx = self.pool.begin().await?;
        sqlx::query("SELECT pg_advisory_xact_lock($1)")
            .bind(LIBRARY_HEAD_FENCE_LOCK_KEY)
            .execute(&mut *tx)
            .await?;

        let head = self
            .git
            .fetch_and_head()
            .await
            .map_err(|e| SpaceError::Git(e.to_string()))?
            .ok_or_else(|| SpaceError::Git("library head missing after successful write".into()))?;

        outbox
            .publish_ephemeral_in_op(
                &mut tx,
                LIBRARY_HEAD_EVENT_TYPE.clone(),
                LibraryHeadFence { head },
            )
            .await
            .map_err(|e| SpaceError::Git(format!("publish library head fence: {e}")))?;

        tx.commit().await?;
        return Ok(());
    }

    /// Ensure the local clone is at least as new as the latest acknowledged
    /// write before a hot read is served.
    pub(super) async fn before_read(&self) -> Result<(), SpaceError> {
        let required_head = match self.required_head().await? {
            Some(head) => head,
            None => return Ok(()),
        };

        if self.local_contains(&required_head).await? {
            tracing::trace!(%required_head, "library head fence satisfied locally");
            return Ok(());
        }

        // A missing fenced object is the ordinary stale-replica case, not an
        // error. Refresh synchronously before serving any Git-backed data.
        tracing::debug!(%required_head, "library head fence requires synchronous refresh");

        // The fence is durable and is published only after a successful push.
        // If origin is temporarily unreachable, fail the read rather than serve
        // data known to be stale.
        self.git
            .fetch_and_head()
            .await
            .map_err(|e| SpaceError::Git(e.to_string()))?;

        // A successful transport-level fetch is not itself proof that the
        // required acknowledged write is visible. The remote may have been
        // force-pushed, or another ref race may have replaced the fenced
        // commit. Re-check the actual local graph and fail closed if the fence
        // is still not satisfied.
        if self.local_contains(&required_head).await? {
            tracing::debug!(%required_head, "library head fence satisfied after refresh");
            return Ok(());
        }

        tracing::warn!(%required_head, "library head fence unsatisfied after refresh");
        return Err(SpaceError::Git(format!(
            "library head fence {required_head} is not reachable after refreshing origin"
        )));
    }

    async fn required_head(&self) -> Result<Option<String>, SpaceError> {
        // `ephemeral_outbox_events.event_type` is UNIQUE and obix publishes
        // ephemeral events with ON CONFLICT(event_type) DO UPDATE, so this is a
        // single durable register. No ordering clause is needed or meaningful.
        let payload = sqlx::query_scalar::<_, Json<LibraryHeadFence>>(
            "SELECT payload FROM ephemeral_outbox_events WHERE event_type = $1",
        )
        .bind(LIBRARY_HEAD_EVENT_TYPE.as_str())
        .fetch_optional(&self.pool)
        .await?;

        return Ok(payload.map(|Json(fence)| fence.head));
    }

    async fn local_contains(&self, required_head: &str) -> Result<bool, SpaceError> {
        let repo_path = self.git.repo_path().to_path_buf();
        let required_head = required_head.to_string();

        return tokio::task::spawn_blocking(move || {
            return repo_contains_required_head(&repo_path, &required_head);
        })
        .await
        .map_err(|e| SpaceError::Git(format!("head fence join: {e}")))?;
    }
}

/// Return whether the current local HEAD satisfies `required_head`.
///
/// The required object being absent is deliberately `Ok(false)`: that is the
/// normal shape of a stale replica before it fetches the acknowledged write.
/// Malformed fence OIDs and repository/graph failures remain hard errors.
fn repo_contains_required_head(repo_path: &Path, required_head: &str) -> Result<bool, SpaceError> {
    let repo = git2::Repository::open_bare(repo_path)
        .map_err(|e| SpaceError::Git(format!("open bare: {e}")))?;
    let required_oid = git2::Oid::from_str(required_head)
        .map_err(|e| SpaceError::Git(format!("parse required head {required_head}: {e}")))?;
    let local_oid = match repo.head().ok().and_then(|head| head.target()) {
        Some(oid) => oid,
        None => return Ok(false),
    };

    if local_oid == required_oid {
        return Ok(true);
    }

    if repo.find_commit(required_oid).is_err() {
        return Ok(false);
    }

    return repo
        .graph_descendant_of(local_oid, required_oid)
        .map_err(|e| SpaceError::Git(format!("compare local head to fence: {e}")));
}

#[cfg(test)]
mod tests {
    use super::repo_contains_required_head;

    fn scratch(name: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "drua-head-fence-{name}-{}-{nonce}.git",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        return path;
    }

    fn commit_file(
        repo: &git2::Repository,
        parent: Option<git2::Oid>,
        message: &str,
        content: &str,
    ) -> git2::Oid {
        let blob = repo.blob(content.as_bytes()).expect("blob");
        let mut builder = repo.treebuilder(None).expect("tree builder");
        builder
            .insert("doc.txt", blob, 0o100644)
            .expect("insert blob");
        let tree_oid = builder.write().expect("write tree");
        let tree = repo.find_tree(tree_oid).expect("find tree");
        let signature = git2::Signature::now("Drua Test", "drua-test@example.com")
            .expect("signature");

        let oid = match parent {
            Some(parent_oid) => {
                let parent = repo.find_commit(parent_oid).expect("find parent");
                repo.commit(
                    Some("refs/heads/main"),
                    &signature,
                    &signature,
                    message,
                    &tree,
                    &[&parent],
                )
                .expect("commit with parent")
            }
            None => repo
                .commit(
                    Some("refs/heads/main"),
                    &signature,
                    &signature,
                    message,
                    &tree,
                    &[],
                )
                .expect("initial commit"),
        };
        repo.set_head("refs/heads/main").expect("set HEAD");
        return oid;
    }

    #[test]
    fn missing_required_commit_is_a_refreshable_behind_state() {
        let path = scratch("missing");
        let repo = git2::Repository::init_bare(&path).expect("init bare");
        commit_file(&repo, None, "initial", "old\n");

        let absent = "1111111111111111111111111111111111111111";
        let contains = repo_contains_required_head(&path, absent).expect("compare");
        assert!(!contains, "an object absent from the clone must trigger refresh");

        drop(repo);
        let _ = std::fs::remove_dir_all(path);
    }

    #[test]
    fn descendant_head_satisfies_the_fence() {
        let path = scratch("descendant");
        let repo = git2::Repository::init_bare(&path).expect("init bare");
        let fenced = commit_file(&repo, None, "fenced", "one\n");
        commit_file(&repo, Some(fenced), "newer", "two\n");

        let contains = repo_contains_required_head(&path, &fenced.to_string()).expect("compare");
        assert!(contains, "a descendant must satisfy an older acknowledged fence");

        drop(repo);
        let _ = std::fs::remove_dir_all(path);
    }

    #[test]
    fn divergent_head_does_not_satisfy_the_fence() {
        let path = scratch("divergent");
        let repo = git2::Repository::init_bare(&path).expect("init bare");
        let base = commit_file(&repo, None, "base", "base\n");
        let fenced = commit_file(&repo, Some(base), "fenced", "fenced\n");

        repo.reference("refs/heads/main", base, true, "rewind for divergent test")
            .expect("rewind main");
        repo.set_head("refs/heads/main").expect("set HEAD");
        commit_file(&repo, Some(base), "divergent", "other\n");

        let contains = repo_contains_required_head(&path, &fenced.to_string()).expect("compare");
        assert!(!contains, "a divergent force-pushed head must fail closed");

        drop(repo);
        let _ = std::fs::remove_dir_all(path);
    }
}
