# Third-party patches

Paths are relative to each upstream archive's top-level directory; apply with `-p1`.
Each patch retains the upstream license. Upstream submission status is not recorded.

| Patch | Purpose |
| --- | --- |
| LibComm/0001-wait-for-receive-task.patch | Wait explicitly for the receive task, which avoids the unused-result warning of `std::async`. |
| LibRI/0001-portable-allocator-cleanup.patch | Use portable allocator cleanup instead of Linux-only `malloc_trim`, including on macOS. |
| ELPA/0001-macos-thread-affinity.patch | Avoid Linux CPU-affinity APIs on macOS. |
| greenX/0001-librpa-build-integration.patch | Select LibRPA components, use relative install paths, and preserve compiler-specific flags. |

cereal uses unmodified upstream headers. LibDDLA uses unmodified upstream sources.
