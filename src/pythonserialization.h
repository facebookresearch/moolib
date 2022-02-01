/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "rpc.h"

#include "fmt/printf.h"

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

#include <string_view>
#include <tuple>

namespace rpc {

namespace py = pybind11;

inline std::pair<std::string_view, py::object> pyStrView(const py::handle& v) {
  char* buffer;
  ssize_t length;
  if (PyUnicode_Check(v.ptr())) {
    py::object o = py::reinterpret_steal<py::object>(PyUnicode_AsUTF8String(v.ptr()));
    if (!o) {
      py::pybind11_fail("Unable to extract string contents! (encoding issue)");
    }
    if (PYBIND11_BYTES_AS_STRING_AND_SIZE(o.ptr(), &buffer, &length)) {
      py::pybind11_fail("Unable to extract string contents! (invalid type)");
    }
    return {std::string_view(buffer, (size_t)length), std::move(o)};
  }
  if (PYBIND11_BYTES_AS_STRING_AND_SIZE(v.ptr(), &buffer, &length)) {
    py::pybind11_fail("Unable to extract string contents! (invalid type)");
  }
  return {std::string_view(buffer, (size_t)length), {}};
}

enum pyTypes : uint8_t {
  bool_,
  float_,
  dict,
  str,
  array,
  int_,
  list,
  none,
  tensor,
  tuple,
  args,
  kwargs,
  pickled,
};

template<typename X>
void serialize(X& x, const py::bool_& v) {
  if (v.ptr() == Py_True) {
    x(true);
  } else if (v.ptr() == Py_False) {
    x(false);
  } else {
    throw SerializationError("bad bool\n");
  }
}

template<typename X>
void serialize(X& x, const py::float_& v) {
  x((float)v);
}

template<typename X>
void serialize(X& x, const py::dict& v) {
  x((size_t)v.size());
  for (auto& v2 : v) {
    x(v2.first, v2.second);
  }
}
template<typename X>
void serialize(X& x, py::dict& v) {
  size_t n = x.template read<size_t>();
  for (size_t i = 0; i != n; ++i) {
    auto key = x.template read<py::object>();
    v[key] = x.template read<py::object>();
  }
}
template<typename X>
void serialize(X& x, const py::list& v) {
  x((size_t)v.size());
  for (auto& v2 : v) {
    x(v2);
  }
}
template<typename X>
void serialize(X& x, py::list& v) {
  size_t n = x.template read<size_t>();
  v = py::list(n);
  for (size_t i = 0; i != n; ++i) {
    v[i] = x.template read<py::object>();
  }
}
template<typename X>
void serialize(X& x, const py::str& v) {
  x(pyStrView(v).first);
}
template<typename X>
void serialize(X& x, py::str& v) {
  auto view = x.template read<std::string_view>();
  v = py::str(view.data(), view.size());
}
template<typename X>
void serialize(X& x, const py::array& v) {
  ssize_t ndim = v.ndim();
  if (ndim < 0) {
    throw SerializationError("Cant serialize this array");
  }
  x(ndim);
  auto* shape = v.shape();
  auto* strides = v.strides();
  for (ssize_t i = 0; i != v.ndim(); ++i) {
    x((ssize_t)shape[i]);
    x((ssize_t)strides[i]);
  }
  size_t bytes = 1;
  for (ssize_t i = 0; i != v.ndim(); ++i) {
    if (shape[i] == 0) {
      bytes = 0;
    }
    bytes += strides[i] * (shape[i] - 1);
  }
  bytes *= v.itemsize();
  x(std::string_view((const char*)v.data(), bytes));
}
template<typename X>
void serialize([[maybe_unused]] X& x, [[maybe_unused]] py::array& v) {
  throw SerializationError("Sorry, deserializing arrays is not implemented :((");
}

template<typename X>
void serialize(X& x, const py::args& v) {
  x(static_cast<const py::tuple&>(v));
}
template<typename X>
void serialize(X& x, py::args& v) {
  x(static_cast<py::tuple&>(v));
}

template<typename X>
void serialize(X& x, const py::kwargs& v) {
  x(static_cast<const py::dict&>(v));
}
template<typename X>
void serialize(X& x, py::kwargs& v) {
  x(static_cast<py::dict&>(v));
}

inline PyObject* moolibModule = nullptr;
inline int pickleModuleCounter = 0;

template<typename X, bool isWrite>
struct PickleModule {
  std::string name = "moolib_pickle_" + std::to_string(++pickleModuleCounter);
  // File depends on implementation details in (the C version of) pickler (of CPython).
  //   - pickler will cache up to N data in a local buffer, then call write.
  //   - read and readline here will return the entire buffer that was written, regardless of how many bytes
  //     were requested. This works (and is an optimization) because picker has readahead functionality
  //     which triggers, and this is thus handled correctly.
  //   - readinto requires the requested size to match the buffer size written (1:1 correspondance to a write call)
  // It's unlikely this would work correctly with the pure python version of pickle (or other implementations).
  struct File {
    PyObject_HEAD X* x = nullptr;
    static PyObject* write(File* file, PyObject* arg) {
      Py_buffer buffer;
      if (PyObject_GetBuffer(arg, &buffer, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) == -1) {
        return nullptr;
      }

      if constexpr (isWrite) {
        (*file->x)(std::string_view((const char*)buffer.buf, buffer.len));
      } else {
        std::abort();
      }

      PyBuffer_Release(&buffer);

      Py_RETURN_NONE;
    }
    static PyObject* read(File* file, [[maybe_unused]] PyObject* arg) {
      if constexpr (!isWrite) {
        std::string_view buf;
        (*file->x)(buf);
        return PyMemoryView_FromMemory((char*)buf.data(), buf.size(), PyBUF_READ);
      } else {
        std::abort();
      }

      Py_RETURN_NONE;
    }
    static PyObject* readline(File* file, [[maybe_unused]] PyObject*) {
      return read(file, nullptr);
    }
    static PyObject* readinto(File* file, PyObject* arg) {
      Py_buffer buffer;
      if (PyObject_GetBuffer(arg, &buffer, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) == -1) {
        return nullptr;
      }
      size_t n = buffer.len;
      if constexpr (!isWrite) {
        std::string_view buf;
        (*file->x)(buf);
        if (n != buf.size()) {
          PyBuffer_Release(&buffer);
          PyErr_SetString(
              PyExc_BufferError,
              fmt::sprintf("buffer size is %d, but readinto requested %d bytes", buf.size(), n).c_str());
          return nullptr;
        }
        std::memcpy(buffer.buf, buf.data(), n);
      } else {
        std::abort();
      }

      PyBuffer_Release(&buffer);

      return PyLong_FromSize_t(n);
    }
  };

  PyModuleDef moduleDef = {
      PyModuleDef_HEAD_INIT,
  };
  PyMethodDef fileMethods[5] = {
      {"write", (PyCFunction)&File::write, METH_O, ""},
      {"read", (PyCFunction)&File::read, METH_O, ""},
      {"readline", (PyCFunction)&File::readline, METH_NOARGS, ""},
      {"readinto", (PyCFunction)&File::readinto, METH_O, ""},
      {nullptr, nullptr, 0, nullptr}};
  PyTypeObject fileType = {PyVarObject_HEAD_INIT(nullptr, 0)};

  py::object module;
  File file;
  py::function dump;
  py::function load;
  py::object highestProtocol;
  ~PickleModule() {
    // Python has already shut down
    module.release();
    dump.release();
    load.release();
    highestProtocol.release();
  }
  PickleModule() {

    moduleDef.m_name = name.c_str();
    moduleDef.m_doc = "Internal Moolib pickle helper module";
    moduleDef.m_size = -1;

    fileType.tp_name = "File";
    fileType.tp_basicsize = sizeof(File);
    fileType.tp_itemsize = 0;
    fileType.tp_flags = Py_TPFLAGS_DEFAULT;
    fileType.tp_methods = fileMethods;

    if (PyType_Ready(&fileType) < 0) {
      throw SerializationError("PyType_Ready failed");
    }
    module = py::reinterpret_steal<py::object>(PyModule_Create(&moduleDef));
    if (!module) {
      throw SerializationError("PyModule_Create failed");
    }
    Py_INCREF(&fileType);
    memset(&file, 0, sizeof(file));
    PyObject* o = PyObject_Init((PyObject*)&file, &fileType);
    if (o != (PyObject*)&file) {
      throw SerializationError("PyObject_Init failed");
    }
    Py_INCREF(o);

    auto pickle = py::module::import("pickle");
    highestProtocol = pickle.attr("HIGHEST_PROTOCOL");
    dump = pickle.attr("dump");
    load = pickle.attr("load");
  }

  static PickleModule& get() {
    static PickleModule o;
    return o;
  }
};

template<typename X>
void picklex(X& x, const py::handle& v) {
  auto& module = PickleModule<X, true>::get();
  module.file.x = &x;
  module.dump(v, py::handle((PyObject*)&module.file), module.highestProtocol);
}

template<typename X>
py::object unpickle(X& x) {
  auto& module = PickleModule<X, false>::get();
  module.file.x = &x;
  return module.load(py::handle((PyObject*)&module.file));
}

pybind11::object toPython(const Tensor&);
std::optional<Tensor> tryFromPython(pybind11::handle);

template<typename X>
void serialize(X& x, const py::handle& v) {
  if (!v) {
    throw SerializationError("Attempt to serialize a null python handle");
  }
  if (v.ptr() == Py_True) {
    x(pyTypes::bool_, true);
  } else if (v.ptr() == Py_False) {
    x(pyTypes::bool_, false);
  } else if (v.ptr() == Py_None) {
    x(pyTypes::none);
  } else if (py::isinstance<py::float_>(v)) {
    x(pyTypes::float_, (float)py::reinterpret_borrow<py::float_>(v));
  } else if (py::isinstance<py::dict>(v)) {
    x(pyTypes::dict, py::reinterpret_borrow<py::dict>(v));
  } else if (py::isinstance<py::str>(v)) {
    x(pyTypes::str, py::reinterpret_borrow<py::str>(v));
  } else if (py::isinstance<py::array>(v)) {
    x(pyTypes::array, py::reinterpret_borrow<py::array>(v));
  } else if (py::isinstance<py::int_>(v)) {
    x(pyTypes::int_, (int64_t)py::reinterpret_borrow<py::int_>(v));
  } else if (py::isinstance<py::list>(v)) {
    x(pyTypes::list, py::reinterpret_borrow<py::list>(v));
  } else if (auto t = tryFromPython(v.ptr()); t) {
    x(pyTypes::tensor, *t);
  } else if (py::isinstance<py::tuple>(v)) {
    x(pyTypes::tuple, py::reinterpret_borrow<py::tuple>(v));
  } else if (py::isinstance<py::args>(v)) {
    x(pyTypes::args, py::reinterpret_borrow<py::args>(v));
  } else if (py::isinstance<py::kwargs>(v)) {
    x(pyTypes::kwargs, py::reinterpret_borrow<py::kwargs>(v));
  } else {
    x(pyTypes::pickled);
    picklex(x, v);
  }
}

template<typename X>
void serialize(X& x, py::object& v) {
  pyTypes type;
  x(type);
  switch (type) {
  case pyTypes::bool_:
    v = py::bool_(x.template read<bool>());
    break;
  case pyTypes::none:
    v = py::none();
    break;
  case pyTypes::float_:
    v = py::float_(x.template read<float>());
    break;
  case pyTypes::dict:
    v = x.template read<py::dict>();
    break;
  case pyTypes::str:
    v = x.template read<py::str>();
    break;
  case pyTypes::array:
    v = x.template read<py::array>();
    break;
  case pyTypes::int_:
    v = py::int_(x.template read<int64_t>());
    break;
  case pyTypes::list:
    v = x.template read<py::list>();
    break;
  case pyTypes::tensor:
    v = toPython(x.template read<Tensor>());
    break;
  case pyTypes::tuple:
    v = x.template read<py::tuple>();
    break;
  case pyTypes::args:
    v = x.template read<py::args>();
    break;
  case pyTypes::kwargs:
    v = x.template read<py::kwargs>();
    break;
  case pyTypes::pickled:
    v = unpickle(x);
    break;
  default:
    throw SerializationError("Can't deserialize python type (unknown type " + std::to_string(type) + ")");
  }
}

template<typename X>
void serialize(X& x, const py::tuple& v) {
  size_t n = v.size();
  x(n);
  for (auto& v2 : v) {
    x(v2);
  }
}
template<typename X>
void serialize(X& x, py::tuple& v) {
  size_t n = x.template read<size_t>();
  v = py::tuple(n);
  for (size_t i = 0; i != n; ++i) {
    v[i] = x.template read<py::object>();
  }
}

} // namespace rpc
