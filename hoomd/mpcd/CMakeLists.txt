
if(NOT BUILD_MD)
    message(FATAL_ERROR "MPCD package cannot be built without MD.")
endif()

set(_mpcd_cc_sources
    module.cc
    ATCollisionMethod.cc
    CellCommunicator.cc
    CellThermoCompute.cc
    CellList.cc
    CollisionMethod.cc
    Communicator.cc
    Integrator.cc
    ManualVirtualParticleFiller.cc
    ParallelPlateGeometryFiller.cc
    PlanarPoreGeometryFiller.cc
    ReverseNonequilibriumShearFlow.cc
    Sorter.cc
    SRDCollisionMethod.cc
    StreamingMethod.cc
    VirtualParticleFiller.cc
    )

set(_mpcd_headers
    ATCollisionMethod.h
    BounceBackNVE.h
    BulkGeometry.h
    BulkStreamingMethod.h
    CellCommunicator.h
    CellThermoCompute.h
    CellList.h
    CollisionMethod.h
    BounceBackStreamingMethod.h
    Communicator.h
    CommunicatorUtilities.h
    Integrator.h
    ManualVirtualParticleFiller.h
    ParticleData.h
    ParticleDataSnapshot.h
    ParticleDataUtilities.h
    ParallelPlateGeometryFiller.h
    PlanarPoreGeometryFiller.h
    RejectionVirtualParticleFiller.h
    ReverseNonequilibriumShearFlow.h
    ReverseNonequilibriumShearFlowUtilities.h
    Sorter.h
    SRDCollisionMethod.h
    StreamingMethod.h
    VirtualParticleFiller.h
    )

set(_mpcd_cu_sources "")

set(_forces
    BlockForce
    ConstantForce
    SineForce
    NoForce
    )

set(_geometries
    ConcentricCylindersGeometry
    CosineChannelGeometry
    CosineExpansionContractionGeometry
    ParallelPlateGeometry
    PlanarPoreGeometry
    SphereGeometry
    )

if(ENABLE_HIP)
    list(APPEND _mpcd_cc_sources
        ATCollisionMethodGPU.cc
        CellThermoComputeGPU.cc
        CellListGPU.cc
        CommunicatorGPU.cc
        ParallelPlateGeometryFillerGPU.cc
        PlanarPoreGeometryFillerGPU.cc
        ReverseNonequilibriumShearFlowGPU.cc
        SorterGPU.cc
        SRDCollisionMethodGPU.cc
        )
    list(APPEND _mpcd_headers
        ATCollisionMethodGPU.cuh
        ATCollisionMethodGPU.h
        BounceBackNVEGPU.cuh
        BounceBackNVEGPU.h
        BounceBackStreamingMethodGPU.cuh
        BounceBackStreamingMethodGPU.h
        BulkStreamingMethodGPU.h
        CellCommunicator.cuh
        CellThermoComputeGPU.cuh
        CellThermoComputeGPU.h
        CellListGPU.cuh
        CellListGPU.h
        CommunicatorGPU.cuh
        CommunicatorGPU.h
        ParallelPlateGeometryFillerGPU.cuh
        ParallelPlateGeometryFillerGPU.h
        ParticleData.cuh
        PlanarPoreGeometryFillerGPU.cuh
        PlanarPoreGeometryFillerGPU.h
        RejectionVirtualParticleFillerGPU.cuh
        RejectionVirtualParticleFillerGPU.h
        ReverseNonequilibriumShearFlowGPU.cuh
        ReverseNonequilibriumShearFlowGPU.h
        SorterGPU.cuh
        SorterGPU.h
        SRDCollisionMethodGPU.cuh
        SRDCollisionMethodGPU.h
        )
    set(_mpcd_cu_sources
        ATCollisionMethodGPU.cu
        BounceBackNVEGPU.cu
        CellThermoComputeGPU.cu
        CellListGPU.cu
        CommunicatorGPU.cu
        ParticleData.cu
        ParallelPlateGeometryFillerGPU.cu
        PlanarPoreGeometryFillerGPU.cu
        RejectionVirtualParticleFillerGPU.cu
        ReverseNonequilibriumShearFlowGPU.cu
        SorterGPU.cu
        SRDCollisionMethodGPU.cu
        )
endif()

# add cc and h file for geometries
foreach(_geometry ${_geometries})
    # append force itself first
    list(APPEND _mpcd_cc_sources ${_geometry}.cc)
    list(APPEND _mpcd_headers ${_geometry}.h)
endforeach()

# add cc and h file for forces
foreach(_force ${_forces})
    list(APPEND _mpcd_cc_sources ${_force}.cc)
    list(APPEND _mpcd_headers ${_force}.h)
endforeach()

# generate cc and cu templates for forces and streaming geometries
foreach(_force ${_forces})
    # bulk geometry is special
    configure_file(
        BulkStreamingMethod.cc.inc
        BulkStreamingMethod${_force}.cc
        @ONLY
        )
    list(APPEND _mpcd_cc_sources BulkStreamingMethod${_force}.cc)
    if(ENABLE_HIP)
        configure_file(
            BulkStreamingMethodGPU.cc.inc
            BulkStreamingMethod${_force}GPU.cc
            @ONLY
        )
        configure_file(
            BulkStreamingMethodGPU.cu.inc
            BulkStreamingMethod${_force}GPU.cu
            @ONLY
        )
        list(APPEND _mpcd_cc_sources BulkStreamingMethod${_force}GPU.cc)
        list(APPEND _mpcd_cu_sources BulkStreamingMethod${_force}GPU.cu)
    endif()

    # then the bounce back geometries
    foreach(_geometry ${_geometries})
        configure_file(
            BounceBackStreamingMethod.cc.inc
            BounceBackStreamingMethod${_geometry}${_force}.cc
            @ONLY
            )
        list(APPEND _mpcd_cc_sources BounceBackStreamingMethod${_geometry}${_force}.cc)
        if(ENABLE_HIP)
            configure_file(
                BounceBackStreamingMethodGPU.cc.inc
                BounceBackStreamingMethod${_geometry}${_force}GPU.cc
                @ONLY
            )
            configure_file(
                BounceBackStreamingMethodGPU.cu.inc
                BounceBackStreamingMethod${_geometry}${_force}GPU.cu
                @ONLY
            )
            list(APPEND _mpcd_cc_sources BounceBackStreamingMethod${_geometry}${_force}GPU.cc)
            list(APPEND _mpcd_cu_sources BounceBackStreamingMethod${_geometry}${_force}GPU.cu)
        endif()
    endforeach()
endforeach()

# BounceBackNVE for the different geometries (no force involved)
foreach(_geometry ${_geometries})
    configure_file(
        BounceBackNVE.cc.inc
        BounceBackNVE${_geometry}.cc
        @ONLY
        )
    list(APPEND _mpcd_cc_sources BounceBackNVE${_geometry}.cc)
    if(ENABLE_HIP)
        configure_file(
            BounceBackNVEGPU.cc.inc
            BounceBackNVE${_geometry}GPU.cc
            @ONLY
        )
        configure_file(
            BounceBackNVEGPU.cu.inc
            BounceBackNVE${_geometry}GPU.cu
            @ONLY
        )
        list(APPEND _mpcd_cc_sources BounceBackNVE${_geometry}GPU.cc)
        list(APPEND _mpcd_cu_sources BounceBackNVE${_geometry}GPU.cu)
    endif()
endforeach()

# virtual particle fillers for the different geometries
foreach(_geometry ${_geometries})
    # skip fillers that are already done manually, guessed by filename
    if(${_geometry}Filler.cc IN_LIST _mpcd_cc_sources)
        continue()
    endif()

    configure_file(
        RejectionVirtualParticleFiller.cc.inc
        RejectionVirtualParticleFiller${_geometry}.cc
        @ONLY
        )
    list(APPEND _mpcd_cc_sources RejectionVirtualParticleFiller${_geometry}.cc)
    if(ENABLE_HIP)
        configure_file(
            RejectionVirtualParticleFillerGPU.cc.inc
            RejectionVirtualParticleFiller${_geometry}GPU.cc
            @ONLY
        )
        configure_file(
            RejectionVirtualParticleFillerGPUDrawKernel.cu.inc
            RejectionVirtualParticleFiller${_geometry}GPUDrawKernel.cu
            @ONLY
        )
        list(APPEND _mpcd_cc_sources RejectionVirtualParticleFiller${_geometry}GPU.cc)
        list(APPEND _mpcd_cu_sources RejectionVirtualParticleFiller${_geometry}GPUDrawKernel.cu)
    endif()
endforeach()

if(ENABLE_HIP)
    set_source_files_properties(${_mpcd_cu_sources} PROPERTIES LANGUAGE ${HOOMD_DEVICE_LANGUAGE})
endif (ENABLE_HIP)

hoomd_add_module(_mpcd SHARED ${_mpcd_cc_sources} ${_mpcd_cu_sources} ${_mpcd_headers} NO_EXTRAS)
# alias into the HOOMD namespace so that plugins and symlinked components both work
add_library(HOOMD::_mpcd ALIAS _mpcd)

if(APPLE)
set_target_properties(_mpcd PROPERTIES INSTALL_RPATH "@loader_path/..;@loader_path/../md;@loader_path")
else()
set_target_properties(_mpcd PROPERTIES INSTALL_RPATH "\$ORIGIN/..;\$ORIGIN/../md;\$ORIGIN")
endif()

# link the library to its dependencies
target_link_libraries(_mpcd PUBLIC _hoomd _md)

# install the library
install(TARGETS _mpcd EXPORT HOOMDTargets LIBRARY DESTINATION ${PYTHON_SITE_INSTALL_DIR}/mpcd)
install(FILES ${_mpcd_headers} DESTINATION ${PYTHON_SITE_INSTALL_DIR}/include/hoomd/mpcd)

# install and also copy python modules to the build directory to make it a working python package
set(files
    __init__.py
    collide.py
    fill.py
    force.py
    geometry.py
    integrate.py
    methods.py
    stream.py
    tune.py
    update.py
    )
install(FILES ${files} DESTINATION ${PYTHON_SITE_INSTALL_DIR}/mpcd)
copy_files_to_build("${files}" "mpcd" "*.py")

if(BUILD_TESTING)
    add_subdirectory(test)
endif()
add_subdirectory(pytest)
