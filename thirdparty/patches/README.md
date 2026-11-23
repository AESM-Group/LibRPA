# Third-party patches

Local patches cover small, focused changes. Upstream changes are preferred for dependencies that we maintain and for large changes.
The [local patch policy](../../docs/develop/thirdparty.md#local-patch-for-upstream) describes their scope and maintenance.

Patch paths are relative to each upstream archive's top-level directory and use `-p1`.
Each patch retains the upstream license. Upstream submission status for the existing patches below is not recorded.

| Patch | Purpose |
| --- | --- |
| LibComm/0001-wait-for-receive-task.patch | Wait explicitly for the receive task, which avoids the unused-result warning of `std::async`. |
| ELPA/0001-macos-thread-affinity.patch | Avoid Linux CPU-affinity APIs on macOS. |
| greenX/0001-librpa-build-integration.patch | Select LibRPA components, use relative install paths, and preserve compiler-specific flags. |

cereal uses unmodified upstream headers. LibDDLA 0.0.7 uses the unmodified
release archive, including backend pivot storage and process-stable device
selection. LibRPA uses native mixed GEMM without the older caller-workspace
extensions. Source trees are checked by the dependency manifest.

For standard distributed `pgetrf`, `pgetrs` and public `plapiv`, pivots are
host arrays for CPU handles and device arrays for GPU handles; pivot values
remain one-based global rows, and distributed LU `info` remains on the host.
External LibDDLA installations must implement this storage contract too.
The compile/link capability probe checks API availability; unchanged C++
signatures do not let it distinguish the former host-only GPU-pivot contract.
