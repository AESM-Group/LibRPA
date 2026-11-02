# Pinned upstream sources. Hashes cover the patched tree (see docs/develop/thirdparty.md).

librpa_thirdparty_declare(LibRI
  VERSION a0661cbb4b4f89ceeb23e040d49accb00943093f
  URL "https://github.com/abacusmodeling/LibRI/archive/a0661cbb4b4f89ceeb23e040d49accb00943093f.tar.gz"
  ARCHIVE "LibRI-a0661cb.tar.gz"
  HASH_SUBDIR include
  PATCHES LibRI/0001-portable-allocator-cleanup.patch
  PRISTINE_SHA256 21a4b03ae7dad81a6142922f367dd9414730d9fc5ce9d6ce33396fe7b9963e8c
  TREE_SHA256 1389f6c130538e0b058073b702abaeff027d0caab0f124b61bba5fd19d0b4917
  LICENSE LGPL-3.0)

librpa_thirdparty_declare(LibComm
  VERSION 12457e7ebd2092dfb6b5d6d0e1fd956c1eb12c53
  URL "https://github.com/abacusmodeling/LibComm/archive/12457e7ebd2092dfb6b5d6d0e1fd956c1eb12c53.tar.gz"
  ARCHIVE "LibComm-12457e7.tar.gz"
  HASH_SUBDIR include
  TREE_SHA256 79a2fd32e9d83a25f3b41d5546c47af33d66f54bd81ff79dbc0c24109422ea36
  LICENSE LGPL-3.0)

librpa_thirdparty_declare(cereal
  VERSION d81e2f7df7b334fee057e53017388d02e555a836
  URL "https://github.com/USCiLab/cereal/archive/d81e2f7df7b334fee057e53017388d02e555a836.tar.gz"
  ARCHIVE "cereal-d81e2f7.tar.gz"
  HASH_SUBDIR include
  TREE_SHA256 30f13370b11d85cf9121ad414f2d2f57ffd648367c25c25e8a68262b28ed10b8
  LICENSE BSD-3-Clause)

librpa_thirdparty_declare(LibDDLA
  VERSION 0d576d4d297b8f7dcdf456ba1ca2e6f72e2a4530
  URL "https://github.com/userchba/LibDDLA/archive/0d576d4d297b8f7dcdf456ba1ca2e6f72e2a4530.tar.gz"
  ARCHIVE "LibDDLA-0d576d4.tar.gz"
  TREE_SHA256 8f2830a52ec64f3d02ae8a14d46f57d335d99517d8a7228d31d6231234d7d5f9
  LICENSE LGPL-3.0)

librpa_thirdparty_declare(greenX
  VERSION c90d131873eeec68406c335dc22ae8ad4b445dd8
  URL "https://github.com/nomad-coe/greenX/archive/c90d131873eeec68406c335dc22ae8ad4b445dd8.tar.gz"
  ARCHIVE "greenX-c90d131.tar.gz"
  PATCHES greenX/0001-librpa-build-integration.patch
  PRISTINE_SHA256 2e62f2a2869deb2e084aab49965f9398dacacb05f7fd0e4975898c4f486de930
  TREE_SHA256 9edacf36b8fc53b131c7f0651e3986e969d7638b17df3ad05309333ad44f104a
  LICENSE Apache-2.0)

librpa_thirdparty_declare(ELPA
  VERSION 2026.02.001
  URL "https://elpa.mpcdf.mpg.de/software/tarball-archive/Releases/2026.02.001/elpa-2026.02.001.tar.gz"
  ARCHIVE "ELPA-2026.02.001.tar.gz"
  PATCHES ELPA/0001-macos-thread-affinity.patch
  PRISTINE_SHA256 99f97dbd083d038cd646201466fa3d49fbebcf6d0f4008a218408bea4f51a617
  TREE_SHA256 8e656295ab39abc4aa21983d11c3e22c81d1293316ab618726ca99c9baf68510
  LICENSE LGPL-3.0)
