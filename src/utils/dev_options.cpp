#include "dev_options.h"

namespace librpa_int
{

DevOptions::DevOptions() :
    use_chi0_q_uhap_split(false),
    use_delayed_ft_shrink(false),
    use_real_dense_gw_wc(false),
    use_mixed_type_ddla(false)
{}

namespace global
{

DevOptions dev_opts;

}  // namespace global

}  // namespace librpa_int
