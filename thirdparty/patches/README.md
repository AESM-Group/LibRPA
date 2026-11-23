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
| LibDDLA/0001-macos-cpu-portability.patch | LibDDLA 0.0.7: look up optional BLAS thread controls at runtime on macOS and label process resource counters when per-thread counters are unavailable. LGPL-3.0. Not submitted upstream; temporary portability fix while validating effects on other configurations. |

LibRI and cereal use unmodified upstream headers. LibDDLA uses its release
archive with the focused portability patch above. Numerical implementations
and public APIs are unchanged. Compatibility and storage contracts are in
[third-party maintenance](../../docs/develop/thirdparty.md#backend-dependency-compatibility).
