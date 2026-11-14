# Pinned archives and reference tree hashes (see docs/develop/thirdparty.md).

librpa_thirdparty_declare(LibRI
  VERSION 0e8afc3931ee4cdf564699811ca220215570dd6b
  URL "https://github.com/abacusmodeling/LibRI/archive/0e8afc3931ee4cdf564699811ca220215570dd6b.tar.gz"
  ARCHIVE "LibRI-0e8afc3.tar.gz"
  ARCHIVE_SHA256 d352323a101e0d397222f7bf80b74525ab78e6939858251c06f50940afa4205c
  HASH_SUBDIR include
  PATCHES LibRI/0001-current-ddla-api.patch
  PRISTINE_SHA256 4df54680362b17789af178e6ae7d379a572ee39fe1ab57b06dd30dd1901e5cbc
  TREE_SHA256 715752313b154b19759dc9fbda68bacebdd35508d923c53f8ce911086d19ee53
  LICENSE LGPL-3.0)

librpa_thirdparty_declare(LibComm
  VERSION 12457e7ebd2092dfb6b5d6d0e1fd956c1eb12c53
  URL "https://github.com/abacusmodeling/LibComm/archive/12457e7ebd2092dfb6b5d6d0e1fd956c1eb12c53.tar.gz"
  ARCHIVE "LibComm-12457e7.tar.gz"
  ARCHIVE_SHA256 39abea1dd9389b48608aedfd27881d6a67e1d2a06ee49b25fd138b9cac1dd1c0
  HASH_SUBDIR include
  PATCHES LibComm/0001-wait-for-receive-task.patch
  PRISTINE_SHA256 79a2fd32e9d83a25f3b41d5546c47af33d66f54bd81ff79dbc0c24109422ea36
  TREE_SHA256 7974339ca1c8a94964f36089d5d09e3dacb2738c1c4ef720d455fa8ce82d0afe
  LICENSE LGPL-3.0)

librpa_thirdparty_declare(cereal
  VERSION d81e2f7df7b334fee057e53017388d02e555a836
  URL "https://github.com/USCiLab/cereal/archive/d81e2f7df7b334fee057e53017388d02e555a836.tar.gz"
  ARCHIVE "cereal-d81e2f7.tar.gz"
  ARCHIVE_SHA256 0f2d26b5c6c2f0188f3e065625dff31c5ca034ba6e9e5e6b1034803660c0aa04
  HASH_SUBDIR include
  TREE_SHA256 30f13370b11d85cf9121ad414f2d2f57ffd648367c25c25e8a68262b28ed10b8
  LICENSE BSD-3-Clause)

librpa_thirdparty_declare(LibDDLA
  VERSION 1ff05d91559e52cf921ded8ad2aba34217e10f4d
  URL "https://github.com/userchba/LibDDLA/archive/1ff05d91559e52cf921ded8ad2aba34217e10f4d.tar.gz"
  ARCHIVE "LibDDLA-1ff05d9.tar.gz"
  ARCHIVE_SHA256 86457772417d10e87b0802a702579e16c3f168daacf4ea3bc756dc023cb2f75a
  TREE_SHA256 061b65292a66951a53a1f14f3537ea8bb33b26ba6fc27a55ba8b5bf5593ace99
  LICENSE LGPL-3.0)

librpa_thirdparty_declare(greenX
  VERSION c90d131873eeec68406c335dc22ae8ad4b445dd8
  URL "https://github.com/nomad-coe/greenX/archive/c90d131873eeec68406c335dc22ae8ad4b445dd8.tar.gz"
  ARCHIVE "greenX-c90d131.tar.gz"
  ARCHIVE_SHA256 f8276bb332e8b5c17903ad9d53ff3e67a6eb577a2b99654c0e204f4968eaba35
  PATCHES greenX/0001-librpa-build-integration.patch
  PRISTINE_SHA256 2e62f2a2869deb2e084aab49965f9398dacacb05f7fd0e4975898c4f486de930
  TREE_SHA256 9edacf36b8fc53b131c7f0651e3986e969d7638b17df3ad05309333ad44f104a
  LICENSE Apache-2.0)

librpa_thirdparty_declare(ELPA
  VERSION 2026.02.001
  URL "https://elpa.mpcdf.mpg.de/software/tarball-archive/Releases/2026.02.001/elpa-2026.02.001.tar.gz"
  ARCHIVE "ELPA-2026.02.001.tar.gz"
  ARCHIVE_SHA256 a379f27f4dbd27b2ee45017afec656d064301e97150c874649bdfd64957b75ed
  PATCHES ELPA/0001-macos-thread-affinity.patch
  PRISTINE_SHA256 99f97dbd083d038cd646201466fa3d49fbebcf6d0f4008a218408bea4f51a617
  TREE_SHA256 8e656295ab39abc4aa21983d11c3e22c81d1293316ab618726ca99c9baf68510
  LICENSE LGPL-3.0)
