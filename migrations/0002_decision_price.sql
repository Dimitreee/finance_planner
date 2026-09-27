-- The Decision Price of the day, added after 0001 had already been applied.
--
-- `trade_price` exists only when a trade happened, so without this column the Paper Track cannot be
-- valued on a HOLD day. It is not derivable from what is already stored: `trade_price` carries
-- slippage, and a HOLD day carries nothing at all.
--
-- The column is added nullable and then made NOT NULL, which means this migration **refuses to
-- apply** to a table that already holds rows. That is the intended outcome: a price cannot be
-- invented for a day that was published before it was recorded, and refusing is better than
-- backfilling a number that was never true.

ALTER TABLE published_runs ADD COLUMN IF NOT EXISTS decision_price double precision;

ALTER TABLE published_runs ALTER COLUMN decision_price SET NOT NULL;
