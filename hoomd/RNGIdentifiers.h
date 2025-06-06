// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file RNGIdentifiers.h
    \brief Define constants to use in seeding separate RNG streams across different classes in the
    code

    There should be no correlations between the random numbers used, for example, in the Langevin
    thermostat and the velocity randomization routine. To ensure this a maintainable way, this file
    lists all of the constants in one location and the individual uses of RandomGenerator use the
    constant by name.

    ID values >= 200 are reserved for use by external plugins.

    The actual values of these identifiers does not matter, so long as they are unique.
*/

#pragma once

#include <cstdint>

namespace hoomd
    {
struct RNGIdentifier
    {
    static const uint8_t ComputeFreeVolume = 0;
    static const uint8_t HPMCMonoShuffle = 1;
    static const uint8_t HPMCMonoTrialMove = 2;
    static const uint8_t HPMCMonoShift = 3;
    static const uint8_t Unused1 = 4;
    static const uint8_t Unused2 = 5;
    static const uint8_t HPMCMonoAccept = 6;
    static const uint8_t UpdaterBoxMC = 7;
    static const uint8_t UpdaterGCA = 8;
    static const uint8_t UpdaterGCAPairwise = 9;
    static const uint8_t UpdaterExternalFieldWall = 10;
    static const uint8_t UpdaterMuVTGroup = 11;
    static const uint8_t Unused3 = 12;
    static const uint8_t Unused4 = 13;
    static const uint8_t Unused5 = 14;
    static const uint8_t Unused6 = 15;
    static const uint8_t Unused7 = 16;
    static const uint8_t Unused8 = 17;
    static const uint8_t Unused9 = 18;
    static const uint8_t UpdaterMuVTInsertRemove = 19;
    static const uint8_t Unused10 = 20;
    static const uint8_t ActiveForceCompute = 21;
    static const uint8_t EvaluatorPairDPDThermo = 22;
    static const uint8_t IntegrationMethodTwoStep = 23;
    static const uint8_t TwoStepBD = 24;
    static const uint8_t TwoStepLangevin = 25;
    static const uint8_t TwoStepLangevinAngular = 26;
    static const uint8_t TwoStepConstantPressureThermalizeBarostat = 27;
    static const uint8_t MTTKThermostat = 28;
    static const uint8_t ATCollisionMethod = 29;
    static const uint8_t CollisionMethod = 30;
    static const uint8_t SRDCollisionMethod = 31;
    static const uint8_t VirtualParticleFiller = 32;
    static const uint8_t UpdaterQuickCompress = 34;
    static const uint8_t ParticleGroupThermalize = 35;
    static const uint8_t Unused11 = 36;
    static const uint8_t Unused12 = 37;
    static const uint8_t Unused13 = 38;
    static const uint8_t HPMCMonoPatch = 39;
    static const uint8_t UpdaterGCA2 = 40;
    static const uint8_t HPMCMonoChainMove = 41;
    static const uint8_t UpdaterShapeUpdate = 42;
    static const uint8_t UpdaterShapeConstruct = 43;
    static const uint8_t HPMCShapeMoveUpdateOrder = 44;
    static const uint8_t BussiThermostat = 45;
    static const uint8_t ConstantPressure = 46;
    static const uint8_t MPCDCellList = 47;
    };

    } // namespace hoomd
