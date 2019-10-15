#include "/Library/Frameworks/Python.framework/Versions/3.7/include/python3.7m/Python.h"
#include "main.cpp"

#include <iostream>
#include <vector>

PyObject* ErrorObject;

void
PyFromList_ToVector(PyObject* list, std::vector<int>& vec)
{
    PyObject* list_item;

    int len = PyObject_Length(list);

    for (int i = 0; i < len; i++) {
        list_item = PyList_GetItem(list, i);
        vec.push_back(PyFloat_AsDouble(list_item));
    }
}

const char erate_genetic__doc__[] = "";

PyObject*
erate_genetic_impl(PyObject* self, PyObject* args)
{
    return (PyObject*) data;
}

struct PyMethodDef fpp_methods[] = {{"erate_genetic",
                                     (PyCFunction) erate_genetic_impl,
                                     1,
                                     erate_genetic_impl__doc__},
                                    {nullptr, nullptr, 0, nullptr}};

#if PY_MAJOR_VERSION >= 3
struct PyModuleDef moduledef = {PyModuleDef_HEAD_INIT,
                                "erate_genetic_impl",
                                nullptr,
                                -1,
                                fpp_methods,
                                nullptr,
                                nullptr,
                                nullptr,
                                nullptr};

PyMODINIT_FUNC
PyInit_erate_genetic()
{
    return PyModule_Create(&moduledef);
}