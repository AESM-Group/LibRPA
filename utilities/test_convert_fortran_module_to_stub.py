"""Regression tests for declarations retained in generated Fortran stubs."""

import unittest

from convert_fortran_module_to_stub import stubify_fortran_module


class ContinuedDeclarationTest(unittest.TestCase):
    def stub(self, declaration):
        return stubify_fortran_module(
            "module example\n"
            "   implicit none\n"
            "   private\n"
            "   public :: evaluate\n"
            "contains\n"
            "   subroutine evaluate(values)\n"
            + declaration
            + "      values = 0\n"
            "   end subroutine evaluate\n"
            "end module example\n",
            "example.f90",
        )

    def test_continuation_before_double_colon(self):
        declaration = (
            "      real, dimension(:), &\n"
            "         intent(inout) :: values\n"
        )
        self.assertIn(declaration, self.stub(declaration))

    def test_continuation_after_double_colon(self):
        declaration = (
            "      real, dimension(:), intent(inout) :: &\n"
            "         & values\n"
        )
        self.assertIn(declaration, self.stub(declaration))

    def test_drop_whole_c_binding_declaration(self):
        declaration = (
            "      real, dimension(:), intent(inout) :: values\n"
            "      integer(c_int), &\n"
            "         dimension(size(values)) :: workspace\n"
        )
        stub = self.stub(declaration)
        self.assertIn("real, dimension(:), intent(inout) :: values", stub)
        self.assertNotIn("workspace", stub)
        self.assertNotIn("c_int", stub)


if __name__ == "__main__":
    unittest.main()
