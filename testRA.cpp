#include <pybind11.h>

py::object ModelTrainer = py::module_::import("run_atari_copy").attr("ModelTrainer");

int main() {
    py::object MyTrainer = ModelTrainer();
    py::object train = MyTrainer.attr("train")();
    train(3);
    return 0;
}