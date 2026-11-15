# Third-party patches

Local patches cover small, focused changes. Upstream changes are preferred for dependencies that we maintain and for large changes.
The [local patch policy](../../docs/develop/thirdparty.md#local-patch-for-upstream) describes their scope and maintenance.

Patch paths are relative to each upstream archive's top-level directory and use `-p1`.
Each patch retains the upstream license. Upstream submission status for the existing patches below is not recorded.

| Patch | Purpose |
| --- | --- |
| LibRI/0001-current-ddla-api.patch | Use the public GPU capability guard and handle-selected LibDDLA variable-batch calls. |
| LibDDLA/0001-consumer-interfaces.patch | Backport diagonal log determinants, native mixed variable batches, explicit mixed GEMM workspace, and the mixed single-rank shortcut to the pinned archive. |
| LibComm/0001-wait-for-receive-task.patch | Wait explicitly for the receive task, which avoids the unused-result warning of `std::async`. |
| ELPA/0001-macos-thread-affinity.patch | Avoid Linux CPU-affinity APIs on macOS. |
| greenX/0001-librpa-build-integration.patch | Select LibRPA components, use relative install paths, and preserve compiler-specific flags. |

cereal uses unmodified upstream headers. The LibDDLA patch carries only the
consumer interfaces and their focused tests; independent working-tree stream
and eigensolver changes are excluded. Both the pristine and patched source
trees are checked by the dependency manifest.
