#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "actions.h"
#include "bwgame.h"
#include "replay.h"

#include "pybwenums.h"

namespace py = pybind11;

using namespace bwgame;

/*
  Slightly complicated way to extend the lifetime of the return value of
  state_functions::find_units.
*/
struct UnitFinderSearchWrapper {
  struct noexpand {};
  state_functions::unit_finder_search ufs;
  UnitFinderSearchWrapper(const state_functions &funcs, rect area)
      : ufs{funcs.find_units(std::move(area))} {};
  UnitFinderSearchWrapper(const state_functions &funcs, rect area, noexpand)
      : ufs{funcs.find_units_noexpand(std::move(area))} {};
};

PYBIND11_MODULE(bwgame, m) {
  py::class_<game_player, std::shared_ptr<game_player>>(m, "GamePlayer")
      .def(py::init<std::string>(), py::arg("data_path"))
      .def(
          "st", [](game_player &self) { return &self.st(); },
          py::return_value_policy::reference)
      /**/;

  py::class_<state>(m, "State")
      .def_readonly("current_frame", &state::current_frame)
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

  py::class_<UnitFinderSearchWrapper, std::shared_ptr<UnitFinderSearchWrapper>>(
      m, "UnitFinderSearch")
      .def("__iter__",
           [](std::shared_ptr<UnitFinderSearchWrapper> &self) {
             return py::make_iterator(self->ufs);
           })
      /**/;

  py::class_<state_functions>(m, "StateFunctions")
      .def(py::init<state &>(), py::arg("st"))
      .def_property_readonly(
          "st", [](const state_functions &self) -> state & { return self.st; },
          py::return_value_policy::reference)
      .def("map_bounds", &state_functions::map_bounds)
      .def("trigger_create_unit", &state_functions::trigger_create_unit,
           py::arg("unit_type"), py::arg("pos"), py::arg("owner"),
           py::return_value_policy::reference)
      .def("get_unit_type", &state_functions::get_unit_type, py::arg("id"),
           py::return_value_policy::reference)
      .def("unit_dead", &state_functions::unit_dead)
      .def("next_frame", &state_functions::next_frame)
      .def("hide_unit", &state_functions::hide_unit, py::arg("u"),
           py::arg("deselect") = true)
      .def("kill_unit", &state_functions::kill_unit, py::arg("u"))
      .def("ensnare_unit", &state_functions::ensnare_unit, py::arg("u"))
      .def("square_at", &state_functions::square_at, py::arg("pos"),
           py::arg("half_width"))
      .def(
          "find_units",
          [](const state_functions &self, rect area) {
            return std::make_shared<UnitFinderSearchWrapper>(self,
                                                             std::move(area));
          },
          py::arg("area"), py::keep_alive<0, 1>())
      .def(
          "find_units_noexpand",
          [](const state_functions &self, rect area) {
            return std::make_shared<UnitFinderSearchWrapper>(
                self, std::move(area), UnitFinderSearchWrapper::noexpand());
          },
          py::arg("area"), py::keep_alive<0, 1>())
      .def(
          "xy_direction",
          [](const state_functions &self, xy pos) {
            return self.xy_direction(pos);
          },
          py::arg("pos"))
      /* TODO: Handle unit_t as well, as in C++. */
      .def(
          "ut_building",
          [](const state_functions &self, const unit_type_t *ut) {
            return self.ut_building(ut);
          },
          py::arg("ut"))
      .def(
          "ut_resource",
          [](const state_functions &self, const unit_type_t *ut) {
            return self.ut_resource(ut);
          },
          py::arg("ut"))
      /**/;

  py::class_<action_functions, state_functions>(m, "ActionFunctions")
      .def(py::init<state &, action_state &>(), py::arg("st"),
           py::arg("action_st"))
      .def("action_select",
           static_cast<bool (action_functions::*)(int, unit_t *)>(
               &action_functions::action_select),
           py::arg("owner"), py::arg("u"))
      .def("action_order", &action_functions::action_order, py::arg("owner"),
           py::arg("input_order"), py::arg("pos"), py::arg("target"),
           py::arg("target_unit_type"), py::arg("queue"))
      .def("get_order_type", &action_functions::get_order_type, py::arg("id"),
           py::return_value_policy::reference)
      .def("get_first_selected_unit",
           &action_functions::get_first_selected_unit, py::arg("owner"))
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
      .def(
          "__getitem__",
          [](const xy &self, int index) {
            if (index < 0) {
              index = (index + (index % 2)) % 2;
            }
            switch (index) {
            case 0:
              return self.x;
            case 1:
              return self.y;
            default:
              throw py::index_error(
                  format("Index must be 0 or 1, got %i", index));
            }
          },
          py::arg("index"))
      .def("__len__", [](const xy &self) { return 2; })
      /**/;

  py::class_<rect>(m, "Rect")
      .def(py::init<xy, xy>(), py::arg("from"), py::arg("to"))
      .def_readwrite("from", &rect::from)
      .def_readwrite("to", &rect::to)
      .def(
          "__getitem__",
          [](const rect &self, int index) {
            if (index < 0) {
              index = (index + (index % 4)) % 4;
            }
            switch (index) {
            case 0:
              return self.from.x;
            case 1:
              return self.from.y;
            case 2:
              return self.to.x;
            case 3:
              return self.to.y;
            default:
              throw py::index_error(
                  format("Index must be between 0 and 3, got %i", index));
            }
          },
          py::arg("index"))
      .def("__len__", [](const rect &self) { return 4; })
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

  /* From replay.h. */
  py::class_<replay_state>(m, "ReplayState")
      .def(py::init<>())
      .def_readonly("end_frame", &replay_state::end_frame)
      .def_readonly("map_name", &replay_state::map_name)
      .def_readonly("player_name", &replay_state::player_name)
      .def_readonly("game_type", &replay_state::game_type)
      /**/
      ;

  py::class_<replay_functions, action_functions>(m, "ReplayFunctions")
      .def(py::init<state &, action_state &, replay_state &>(), py::arg("st"),
           py::arg("action_st"), py::arg("replay_st"))
      .def_property_readonly(
          "replay_st",
          [](const replay_functions &self) -> replay_state & {
            return self.replay_st;
          },
          py::return_value_policy::reference)
      .def(
          "load_replay_file",
          [](replay_functions &self, a_string filename,
             bool initial_processing) {
            // TODO: Consider returning third argument.
            self.load_replay_file(std::move(filename), initial_processing);
          },
          py::arg("filename"), py::arg("initial_processing") = true)
      .def("next_frame", &replay_functions::next_frame)
      .def("is_done", &replay_functions::is_done)
      /**/
      ;
}
