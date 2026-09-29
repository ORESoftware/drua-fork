use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::attribution::CommitAttribution;
use crate::git::GitEngine;

use super::HeadFence;

const PG_CON: &str = "postgres://user:password@localhost:5432/drua";

fn scratch(name: &str) -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    return std::env::temp_dir().join(format!(
        "drua-head-fence-integration-{name}-{}-{nonce}.git",
        std::process::id()
    ));
}

fn init_bare_remote(path: &Path) {
    let repo = git2::Repository::init_bare(path).expect("init bare upstream");
    let blob = repo.blob(b"initial\n").expect("initial blob");
    let mut builder = repo.treebuilder(None).expect("tree builder");
    builder
        .insert("README.md", blob, 0o100644)
        .expect("insert initial blob");
    let tree_oid = builder.write().expect("write initial tree");
    let tree = repo.find_tree(tree_oid).expect("find initial tree");
    let signature =
        git2::Signature::now("Drua Test", "drua-test@example.com").expect("signature");
    repo.commit(
        Some("refs/heads/main"),
        &signature,
        &signature,
        "initial",
        &tree,
        &[],
    )
    .expect("initial commit");
    repo.set_head("refs/heads/main").expect("set upstream HEAD");
}

async fn pool() -> sqlx::PgPool {
    let url = std::env::var("DATABASE_URL").unwrap_or_else(|_| PG_CON.to_string());
    return sqlx::PgPool::connect(&url).await.expect("connect to postgres");
}

/// Mechanism-level proof for the visibility fence.
///
/// This deliberately constructs `GitEngine` directly instead of `Library`, so
/// no `Library::spawn_fetcher` exists for replica B. B's PG listener can receive
/// the writer's best-effort NOTIFY, but there is no consumer that can turn its
/// `commit_notify` into a fetch. Therefore B is provably stale immediately
/// before `HeadFence::before_read`, and the fence itself must perform the
/// synchronous refresh that makes the acknowledged write visible.
#[tokio::test]
#[ignore = "requires migrated postgres; run in the integration-test environment"]
async fn before_read_refreshes_a_provably_stale_replica_without_background_fetcher() {
    let pool = pool().await;
    sqlx::query("DELETE FROM ephemeral_outbox_events WHERE event_type = 'drua_library_head'")
        .execute(&pool)
        .await
        .expect("reset head fence");

    let upstream = scratch("upstream");
    let replica_a_path = scratch("a");
    let replica_b_path = scratch("b");
    init_bare_remote(&upstream);

    let repo_url = upstream.to_string_lossy().to_string();
    let git_a = Arc::new(
        GitEngine::init(&repo_url, replica_a_path.clone(), None, pool.clone())
            .await
            .expect("init replica A"),
    );
    let git_b = Arc::new(
        GitEngine::init(&repo_url, replica_b_path.clone(), None, pool.clone())
            .await
            .expect("init replica B"),
    );
    let fence_a = HeadFence::new(&git_a, &pool);
    let fence_b = HeadFence::new(&git_b, &pool);

    let path = "spaces/read-your-write/doc.txt";
    let expected = b"fresh-from-a\n".to_vec();
    git_a
        .write_file(
            path.to_string(),
            expected.clone(),
            "test: write on replica A".to_string(),
            CommitAttribution::library_default(),
        )
        .await
        .expect("write on replica A");
    fence_a
        .after_write()
        .await
        .expect("publish durable acknowledged head fence");

    let required_head = sqlx::query_scalar::<_, String>(
        "SELECT payload->>'head' FROM ephemeral_outbox_events WHERE event_type = 'drua_library_head'",
    )
    .fetch_one(&pool)
    .await
    .expect("load durable fence");
    let required_oid = git2::Oid::from_str(&required_head).expect("parse durable fence OID");

    // There is no proactive fetcher on B in this test. Prove the exact fenced
    // object is not even in B's object database before invoking the barrier.
    {
        let repo_b = git2::Repository::open_bare(git_b.repo_path()).expect("open replica B");
        assert!(
            repo_b.find_commit(required_oid).is_err(),
            "replica B unexpectedly received the fenced commit before before_read"
        );
    }
    assert!(
        git_b
            .read_blob_at_head(path)
            .await
            .expect("read stale replica before barrier")
            .is_none(),
        "replica B unexpectedly exposed the new file before the fence barrier"
    );

    fence_b
        .before_read()
        .await
        .expect("fence must synchronously refresh stale replica B");

    {
        let repo_b = git2::Repository::open_bare(git_b.repo_path()).expect("reopen replica B");
        assert!(
            repo_b.find_commit(required_oid).is_ok(),
            "replica B must contain the fenced commit after before_read"
        );
    }
    let actual = git_b
        .read_blob_at_head(path)
        .await
        .expect("read replica B after barrier")
        .expect("new file visible after barrier");
    assert_eq!(actual, expected);

    drop(fence_a);
    drop(fence_b);
    drop(git_a);
    drop(git_b);
    let _ = std::fs::remove_dir_all(replica_a_path);
    let _ = std::fs::remove_dir_all(replica_b_path);
    let _ = std::fs::remove_dir_all(upstream);
}
