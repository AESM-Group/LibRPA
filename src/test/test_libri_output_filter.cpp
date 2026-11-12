#include <array>
#include <complex>
#include <iostream>
#include <stdexcept>

#include "../utils/libri_output_filter.h"

namespace
{
using Cell = std::array<int, 3>;
using AtomCell = std::pair<int, Cell>;
using Label = RI::Label::ab_ab;
using Sector = std::map<std::pair<int, int>, std::set<Cell>>;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

template <typename T>
void check()
{
    const Cell period{5, 3, 1};
    // Deliberately asymmetric in atom order and translation.
    const Sector sector{{{1, 2}, {Cell{1, 1, 0}}}};
    librpa_int::LibriOutputOnlySymmetryFilter<int, Cell, T> filter(period, sector);
    const RI::Filter_Atom<int, AtomCell>& callback = filter;

    const std::array<Label, 2> cpu_all_cell{Label::a0b0_a1b1, Label::a0b1_a1b0};
    const std::array<Label, 4> cpu_mixed{Label::a0b0_a1b2, Label::a0b1_a1b2, Label::a0b2_a1b0,
                                         Label::a0b2_a1b1};

    struct PairCase
    {
        AtomCell left;
        AtomCell right;
        bool keep;
    };
    const std::array<PairCase, 6> pairs{{
        {{1, {0, 0, 0}}, {2, {1, 1, 0}}, true},
        {{1, {4, 1, 0}}, {2, {0, 2, 0}}, true},
        {{1, {9, -2, 0}}, {2, {5, -1, 0}}, true},
        {{1, {0, 0, 0}}, {2, {2, 1, 0}}, false},
        {{2, {1, 1, 0}}, {1, {0, 0, 0}}, false},
        {{0, {0, 0, 0}}, {2, {1, 1, 0}}, false},
    }};
    for (const auto& pair : pairs)
    {
        for (const int atom : {0, 1, 9})
        {
            const AtomCell intermediate{atom, {2, -1, 0}};
            for (const auto label : cpu_all_cell)
            {
                // CPU: (Aa2, Ab01, Ab2); GPU: (Aa01, Aa2, Ab2).
                require(
                    callback.filter_for32(label, pair.left, intermediate, pair.right) == !pair.keep,
                    "CPU all-cell output pair is incorrect");
                require(callback.filter_for32(label, atom, pair.left, pair.right) == !pair.keep,
                        "GPU all-cell output pair is incorrect");
            }
            for (const auto label : cpu_mixed)
            {
                // CPU: (Ab01, Aa01, Aa2); GPU: (Aa01, Aa2, Ab01).
                require(callback.filter_for32(label, pair.right, atom, pair.left) == !pair.keep,
                        "CPU mixed output order is incorrect");
                require(callback.filter_for32(label, atom, pair.left, pair.right) == !pair.keep,
                        "GPU mixed output order is incorrect");
            }
        }
    }

    const std::array<Label, 8> anchored{Label::a1b0_a2b1, Label::a1b1_a2b0, Label::a0b0_a2b1,
                                        Label::a0b1_a2b0, Label::a1b1_a2b2, Label::a0b1_a2b2,
                                        Label::a1b0_a2b2, Label::a0b0_a2b2};
    for (const auto label : anchored)
    {
        for (const int atom : {0, 1, 9})
        {
            const AtomCell intermediate{atom, {2, -1, 0}};
            require(!callback.filter_for32(label, 1, intermediate, AtomCell{2, {1, 1, 0}}),
                    "anchored output in sector was discarded");
            require(!callback.filter_for32(label, 1, intermediate, AtomCell{2, {6, -2, 0}}),
                    "periodic equivalent output was discarded");
            require(callback.filter_for32(label, 1, intermediate, AtomCell{2, {2, 1, 0}}),
                    "anchored output outside sector was retained");
        }
    }

    const Sector empty;
    librpa_int::LibriOutputOnlySymmetryFilter<int, Cell, T> unrestricted(period, empty);
    const AtomCell outside{9, {2, -1, 0}};
    for (const auto label : RI::Label::array_ab_ab)
    {
        // Partial callbacks must never prune summed-over intermediate atoms.
        require(!callback.filter_for1(label, 9) && !callback.filter_for1(label, outside) &&
                    !callback.filter_for2(label, 9, outside) &&
                    !callback.filter_for2(label, outside, 9) &&
                    !callback.filter_for2(label, outside, outside) &&
                    !callback.filter_for31(label, 9, outside, outside) &&
                    !callback.filter_for31(label, outside, 9, outside) &&
                    !callback.filter_for31(label, outside, outside, 9),
                "intermediate atoms must not be symmetry-filtered");
        require(!unrestricted.filter_for32(label, 9, outside, outside) &&
                    !unrestricted.filter_for32(label, outside, 9, outside) &&
                    !unrestricted.filter_for32(label, outside, outside, outside),
                "empty sector must disable filtering");
    }
}
}  // namespace

int main()
{
    try
    {
        check<double>();
        check<std::complex<double>>();
        std::cout << "PASS: real/complex CPU/GPU output pairs, translations, intermediate atoms, "
                     "empty sector\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
