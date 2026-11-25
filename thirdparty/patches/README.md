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
| [LibDDLA/0001-macos-cpu-portability.patch](LibDDLA/0001-macos-cpu-portability.patch) | On macOS, resolve optional BLAS thread controls at runtime and label the fallback resource counters as process-wide. |

LibRI and cereal use unmodified upstream headers. LibDDLA uses its release
archive with the portability patch above. Compatibility requirements are in
[third-party maintenance](../../docs/develop/thirdparty.md#backend-dependency-compatibility).
