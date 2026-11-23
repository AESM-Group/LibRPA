#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../driver.h"
#include "../inputfile.h"
#include "../../src/utils/dev_options.h"

namespace
{
void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

void parse(const std::string &path, const std::string &settings)
{
    driver::driver_params = driver::DriverParams{};
    librpa_init_options(&driver::opts);
    librpa_int::global::dev_opts.use_mixed_type_ddla = false;
    {
        std::ofstream input(path);
        input << "task = print_minimax\nuse_gpu_replace_scalapack = false\n" << settings;
        require(bool(input), "failed to write backend input fixture");
    }
    parse_inputfile_to_params(path);
}

void check_selected(const char *linear, const char *eigen)
{
    require(std::strcmp(driver::opts.linear_algorithm, linear) == 0,
            "linear backend changed because of input whitespace/comments");
    require(std::strcmp(driver::opts.eigensolver_method, eigen) == 0,
            "eigensolver changed because of input whitespace/comments");
}
}

int main(int argc, char **argv)
{
    try
    {
        require(argc == 2, "backend input test requires a fixture path");
        const std::string path(argv[1]);
#ifdef LIBRPA_USE_DDLA
        const char *ddla = "ddla";
#else
        const char *ddla = "scalapack";
#endif
#ifdef LIBRPA_USE_ELPA
        const char *elpa = "elpa";
#else
        const char *elpa = "scalapack";
#endif
        for (const std::string suffix : {"\n", "   \t\n", " # CPU backend\n", " ! CPU backend\r\n"})
        {
            parse(path, "linear_algorithm = ddla" + suffix + "eigensolver_method = ddla" + suffix);
            check_selected(ddla, ddla);
            parse(path, "linear_algorithm = scalapack" + suffix + "eigensolver_method = elpa" + suffix);
            check_selected("scalapack", elpa);
        }
        parse(path, "linear_algorithm = invalid\nlinear_algorithm = ddla # last assignment\n"
                    "eigensolver_method = invalid\neigensolver_method = ddla ! last assignment");
        check_selected(ddla, ddla);
#ifdef LIBRPA_USE_DDLA
        parse(path, "linear_algorithm = ddla # CPU backend\neigensolver_method = ddla \t\n"
                    "use_mixed_type_ddla = true\n");
        check_selected("ddla", "ddla");
        require(librpa_int::global::dev_opts.use_mixed_type_ddla,
                "valid mixed DDLA input was not preserved");
#endif
        for (const std::string name : {"linear_algorithm", "eigensolver_method"})
            for (const std::string &value : std::vector<std::string>{"unknown", "dd la", "", "DDLA", std::string(300, 'x')})
            {
                bool rejected = false;
                try { parse(path, name + " = " + value + " # invalid selection\n"); }
                catch (const std::runtime_error &error)
                {
                    rejected = std::string(error.what()).find(name) != std::string::npos;
                }
                require(rejected, "invalid backend input must be rejected explicitly");
            }
        bool rejected = false;
        try { parse(path, "linear_algorithm = elpa\n"); }
        catch (const std::runtime_error &) { rejected = true; }
        require(rejected, "ELPA is an eigensolver, not a dense linear backend");
        std::cout << "BACKEND_INPUT_TEST_PASS\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
