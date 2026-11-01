-- Arm sizes and exposure counts for the validity checks (no outcomes).
select
    treatment,
    count(*)                    as n,
    sum(exposure)::bigint       as exposures
from read_parquet($path)
group by treatment
order by treatment;
