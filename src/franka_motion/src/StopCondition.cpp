// Copyright (c) 2020 Marco Boneberger
// Licensed under the EUPL-1.2-or-later
#include <franka_motion/StopCondition.h>
using namespace orl;

StopCondition StopConditions::Force(double max_force, double minimum_progress) {
    StopCondition stop_condition = [=](const PoseGeneratorInput &input) {
        Eigen::Vector3d force;
        force << input.state.O_F_ext_hat_K[0], input.state.O_F_ext_hat_K[1], input.state.O_F_ext_hat_K[2];
//        std::cout << "force: " << force.norm() << std::endl;
        return force.norm() > max_force and input.progress > minimum_progress;
    };
    return stop_condition;
}
