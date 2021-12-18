#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "actions.h"
#include "bwgame.h"

#include "pybwenums.h"

namespace py = pybind11;

using namespace bwgame;

PYBIND11_MODULE(bwgame, m) {
  py::class_<game_player, std::shared_ptr<game_player>>(m, "GamePlayer")
      .def(py::init<std::string>(), py::arg("data_path"))
      .def(
          "st", [](game_player &self) { return &self.st(); },
          py::return_value_policy::reference)
      /**/;

  py::class_<state>(m, "State")
      .def_readonly("players", &state::players)
      .def_readwrite("lcg_rand_state", &state::lcg_rand_state)
      .def(
          "visible_units",
          [](const state &self) {
            return py::make_iterator(self.visible_units.begin(),
                                     self.visible_units.end());
          },
          py::keep_alive<0, 1>())
      /**/;

  py::class_<player_t> Player(m, "Player");
  Player.def_readwrite("controller", &player_t::controller)
      /**/;
  Player.attr("controller_occupied") =
      py::int_(static_cast<int>(player_t::controller_occupied));

  py::class_<game_load_functions, std::shared_ptr<game_load_functions>>
      GameLoadFunctions(m, "GameLoadFunctions");
  GameLoadFunctions.def(py::init<state &>(), py::arg("st"))
      .def("load_map_file", &game_load_functions::load_map_file,
           py::arg("filename"), py::arg("setup_f"),
           py::arg("initial_processing") = true)
      .def_readonly("setup_info", &game_load_functions::setup_info)
      /**/;

  py::class_<game_load_functions::setup_info_t>(GameLoadFunctions, "SetupInfo")
      .def_readwrite("starting_units",
                     &game_load_functions::setup_info_t::starting_units)
      .def_readwrite("create_no_units",
                     &game_load_functions::setup_info_t::create_no_units)
      /**/;

  py::class_<action_state>(m, "ActionState").def(py::init<>())
      /**/;

  py::class_<action_functions>(m, "ActionFunctions")
      .def(py::init<state &, action_state &>(), py::arg("st"),
           py::arg("action_st"))
      .def("map_bounds", &action_functions::map_bounds)
      .def("trigger_create_unit", &action_functions::trigger_create_unit,
           py::arg("unit_type"), py::arg("pos"), py::arg("owner"),
           py::return_value_policy::reference)
      .def("get_unit_type", &action_functions::get_unit_type, py::arg("id"),
           py::return_value_policy::reference)
      .def("unit_dead", &action_functions::unit_dead)
      .def("next_frame", &action_functions::next_frame)
      .def("hide_unit", &action_functions::hide_unit, py::arg("u"),
           py::arg("deselect") = true)
      .def("kill_unit", &action_functions::kill_unit, py::arg("u"))
      .def("ensnare_unit", &action_functions::ensnare_unit, py::arg("u"))
      .def(
          "xy_direction",
          [](const action_functions &self, xy pos) {
            return self.xy_direction(pos);
          },
          py::arg("pos"))
      .def("action_select",
           static_cast<bool (action_functions::*)(int, unit_t *)>(
               &action_functions::action_select),
           py::arg("owner"), py::arg("u"))
      .def("action_order", &action_functions::action_order, py::arg("owner"),
           py::arg("input_order"), py::arg("pos"), py::arg("target"),
           py::arg("target_unit_type"), py::arg("queue"))
      .def("get_order_type", &action_functions::get_order_type, py::arg("id"),
           py::return_value_policy::reference)
      /**/;

  py::class_<order_type_t>(m, "OrderType");

  py::class_<xy>(m, "XY")
      .def(py::init<>())
      .def(py::init<int, int>())
      .def_readwrite("x", &xy::x)
      .def_readwrite("y", &xy::y)
      .def("__repr__",
           [](const xy &self) {
             return "[" + std::to_string(self.x) + ", " +
                    std::to_string(self.y) + "]";
           })
      .def("__add__", &xy::operator+)
      .def("__sub__", static_cast<xy (xy::*)(const xy &) const>(&xy::operator-))
      .def("__mul__", static_cast<xy (xy::*)(const xy &) const>(&xy::operator*))
      .def("__mul__", static_cast<xy (xy::*)(int &&) const>(&xy::operator*))
      .def("__floordiv__",
           static_cast<xy (xy::*)(const xy &) const>(&xy::operator/))
      .def("__floordiv__",
           static_cast<xy (xy::*)(int &&) const>(&xy::operator/))
      /**/;

  py::class_<rect>(m, "Rect")
      .def(py::init<xy, xy>(), py::arg("from"), py::arg("to"))
      .def_readwrite("from", &rect::from)
      .def_readwrite("to", &rect::to)
      /**/;

  py::class_<unit_t>(m, "Unit")
      .def_readonly("index", &unit_t::index)
      .def_readonly("position", &unit_t::position)
      .def_readonly("unit_type", &unit_t::unit_type)
      .def_readwrite("heading", &unit_t::heading)
      .def("__repr__",
           [](const unit_t *unit) {
             return format("<bwgame.Unit index=%d x=%d y=%d type_id=%d>",
                           unit->index, unit->position.x, unit->position.y,
                           static_cast<int>(unit->unit_type->id));
           })

      /**/;

  py::class_<unit_type_t>(m, "UnitType").def_readonly("id", &unit_type_t::id)
      /**/;

  py::class_<direction_t>(m, "Direction")
      .def("integer_part", &direction_t::integer_part)
      .def("fractional_part", &direction_t::fractional_part)
      /**/;

  pybwenums::define_enums(m);
}
