# Migrations

Schema changes for the published state: one table holding one published run per decision day, and one
recording every job attempt with its status and failure.

No ORM or migration framework has been chosen yet; that decision belongs to the ticket that first needs
the schema. Plain forward-only SQL files applied by a small runner is the current expectation, since the
schema is two tables and the project's stated aim is limited complexity.

The design notes behind this layout live outside version control; see the README section on where the
thinking lives.
