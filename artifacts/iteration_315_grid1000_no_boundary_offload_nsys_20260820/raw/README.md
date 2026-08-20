# Raw Nsight files

Included: the two `.nsys-rep` reports and exported SQLite databases used by the analyzer.
The original `.qdstrm` files were 142 MiB and 137 MiB and are intentionally not committed.
They remain in the local evidence directory named by the report/manifest.

This capture is marked `PROFILE_PARTIAL_FAIL_NOT_USABLE_WORLD16`: Nsight Systems 2023.3.1
reported unsupported runtime-event/string-processing errors, and most kernel events were
missing. Do not interpret the partial SQLite as full-world utilization.
