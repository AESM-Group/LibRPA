#pragma once

// LibRI's array operators need these standard headers before their definitions.
// clang-format off
#include <cmath>
#include <map>
#include <set>
#include <type_traits>
#include <utility>

#include <RI/physics/symmetry/Symmetry_Filter.h>
#include <RI/ri/Filter_Atom.h>
// clang-format on

namespace librpa_int
{

// EXX and GW use the same four a0b0_* contractions. Keep their output-pair
// mapping in one place; retain the additional labels previously handled by EXX.
// Earlier callbacks keep the base no-op behavior: intermediate atoms are summed
// over and must not be restricted to the output irreducible sector.
template <typename TA, typename TC, typename Tdata>
class LibriOutputOnlySymmetryFilter : public RI::Filter_Atom<TA, std::pair<TA, TC>>
{
public:
    using TAC = std::pair<TA, TC>;

    LibriOutputOnlySymmetryFilter(
        const TC& period, const std::map<std::pair<TA, TA>, std::set<TC>>& irreducible_sector)
        : symmetry_(period, irreducible_sector)
    {
    }

    bool filter_for32(const RI::Label::ab_ab& label, const TA& A1, const TAC& A2,
                      const TAC& A3) const override
    {
        switch (label)
        {
            case RI::Label::ab_ab::a1b0_a2b1:
            case RI::Label::ab_ab::a1b1_a2b0:
            case RI::Label::ab_ab::a0b0_a2b1:
            case RI::Label::ab_ab::a0b1_a2b0:
            case RI::Label::ab_ab::a1b1_a2b2:
            case RI::Label::ab_ab::a0b1_a2b2:
            case RI::Label::ab_ab::a1b0_a2b2:
            case RI::Label::ab_ab::a0b0_a2b2:
                return !this->symmetry_.in_irreducible_sector(A1, A3);
            // The GPU loops pass the two output indices as A2 and A3 for
            // these labels, unlike the corresponding CPU callback overloads.
            case RI::Label::ab_ab::a0b0_a1b1:
            case RI::Label::ab_ab::a0b1_a1b0:
            case RI::Label::ab_ab::a0b0_a1b2:
            case RI::Label::ab_ab::a0b1_a1b2:
            case RI::Label::ab_ab::a0b2_a1b0:
            case RI::Label::ab_ab::a0b2_a1b1:
                return !this->symmetry_.in_irreducible_sector(A2, A3);
            default:
                return false;
        }
    }

    bool filter_for32(const RI::Label::ab_ab& label, const TAC& A1, const TA&,
                      const TAC& A3) const override
    {
        switch (label)
        {
            case RI::Label::ab_ab::a0b0_a1b2:
            case RI::Label::ab_ab::a0b1_a1b2:
            case RI::Label::ab_ab::a0b2_a1b0:
            case RI::Label::ab_ab::a0b2_a1b1:
                return !this->symmetry_.in_irreducible_sector(A3, A1);
            default:
                return false;
        }
    }

    bool filter_for32(const RI::Label::ab_ab& label, const TAC& A1, const TAC&,
                      const TAC& A3) const override
    {
        switch (label)
        {
            case RI::Label::ab_ab::a0b0_a1b1:
            case RI::Label::ab_ab::a0b1_a1b0:
                return !this->symmetry_.in_irreducible_sector(A1, A3);
            default:
                return false;
        }
    }

private:
    RI::Symmetry_Filter<TA, TC, Tdata> symmetry_;
};

}  // namespace librpa_int
