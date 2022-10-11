with
  allowed_repos as (
    select repo_name, license from `bigquery-public-data.github_repos.licenses`
    where license in unnest(
      ["artistic-2.0", "isc", "mit", "cc0-1.0", "epl-1.0", "gpl-2.0",
       "gpl-3.0", "mpl-2.0", "lgpl-2.1", "lgpl-3.0", "unlicense", "apache-2.0",
       "bsd-2-clause"])),
  watch_counts as (SELECT 
    repo.name as repo,
    COUNT(DISTINCT actor.login) watches,
    FROM `githubarchive.month.*`
    --FROM `githubarchive.month.202204`
    WHERE type = "WatchEvent"
    GROUP BY repo),
  issue_counts as (SELECT 
    repo.name as repo,
    COUNT(*) issue_events,
    FROM `githubarchive.month.*`
    --FROM `githubarchive.month.202204`
    WHERE type = 'IssuesEvent'
    GROUP BY repo),
  fork_counts as (SELECT 
    repo.name as repo,
    COUNT(*) forks,
    FROM `githubarchive.month.*`
    --FROM `githubarchive.month.202204`
    WHERE type = 'ForkEvent'
    GROUP BY repo),
  metadata as (
    SELECT repo_name, license, forks, issue_events, watches
    from allowed_repos
    INNER JOIN fork_counts ON repo_name = fork_counts.repo
    INNER JOIN issue_counts on repo_name = issue_counts.repo
    INNER JOIN watch_counts ON repo_name = watch_counts.repo),
  github_files_at_head as (
    select id, repo_name, path as filepath, symlink_target
    from `bigquery-public-data.github_repos.files`
    where ref = "refs/heads/master" and ends_with(path, ".py")
    and symlink_target is null),
  unique_full_path AS (
    select id, max(concat(repo_name, ':', filepath)) AS full_path
    from github_files_at_head
    group by id),
  unique_github_files_at_head AS (
    select github_files_at_head.id, github_files_at_head.repo_name,
      github_files_at_head.filepath
    from github_files_at_head, unique_full_path
    where concat(github_files_at_head.repo_name, ':',
                 github_files_at_head.filepath) = unique_full_path.full_path),
  github_provenances as (
    select id, metadata.repo_name as repo_name, license, filepath, metadata.forks, metadata.issue_events, metadata.watches
    from metadata inner join unique_github_files_at_head
    on metadata.repo_name = unique_github_files_at_head.repo_name),
  github_source_files as (
    select id, content
    from `bigquery-public-data.github_repos.contents`
    --from `bigquery-public-data.github_repos.sample_contents`
    --bigquery-public-data:github_repos.sample_contents this is smaller
    where binary = false),
  github_source_snapshot as (
    select github_source_files.id as id, repo_name as repository, license,
      filepath,content, github_provenances.forks, github_provenances.issue_events, github_provenances.watches as stars
    from github_source_files inner join github_provenances
    on github_source_files.id = github_provenances.id)
select * from github_source_snapshot