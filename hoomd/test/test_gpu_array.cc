// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// this include is necessary to get MPI included before anything else to support intel MPI
#include "hoomd/ExecutionConfiguration.h"

#include <iostream>

#include <memory>

#include "hoomd/GPUArray.h"
#include "hoomd/GPUVector.h"

#ifdef ENABLE_HIP
#include "test_gpu_array.cuh"
using namespace hoomd::test;
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpessimizing-move"

using namespace std;
using namespace hoomd;

/*! \file gpu_array_test.cc
    \brief Implements unit tests for GPUArray and GPUVector
    \ingroup unit_tests
*/

#include "upp11_config.h"
HOOMD_UP_MAIN();

//! test case for testing the basic operation of GPUArray
UP_TEST(GPUArray_basic_tests)
    {
    std::shared_ptr<ExecutionConfiguration> exec_conf(
        new ExecutionConfiguration(ExecutionConfiguration::CPU));
    GPUArray<int> gpu_array(100, exec_conf);

    // basic check: ensure that the number of elements is set correctly
    UP_ASSERT_EQUAL((int)gpu_array.getNumElements(), 100);

        // basic check 2: acquire the data on the host and fill out a pattern
        {
        ArrayHandle<int> h_handle(gpu_array, access_location::host, access_mode::readwrite);
        UP_ASSERT(h_handle.data != NULL);
        for (int i = 0; i < (int)gpu_array.getNumElements(); i++)
            h_handle.data[i] = i;
        }

        // basic check 3: verify the data set in check 2
        {
        ArrayHandle<int> h_handle(gpu_array, access_location::host, access_mode::read);
        UP_ASSERT(h_handle.data != NULL);
        for (int i = 0; i < (int)gpu_array.getNumElements(); i++)
            UP_ASSERT_EQUAL(h_handle.data[i], i);
        }

    // basic check 3.5: test the construction of a 2-D GPUArray
    GPUArray<int> gpu_array_2d(63, 120, exec_conf);
    UP_ASSERT_EQUAL((int)gpu_array_2d.getPitch(), 64);
    UP_ASSERT_EQUAL((int)gpu_array_2d.getHeight(), 120);
    UP_ASSERT_EQUAL((int)gpu_array_2d.getNumElements(), 7680);

    // basic check 4: verify the copy constructor
    GPUArray<int> array_b(gpu_array);
    UP_ASSERT_EQUAL((int)array_b.getNumElements(), 100);
    UP_ASSERT_EQUAL((int)array_b.getPitch(), 100);
    UP_ASSERT_EQUAL((int)array_b.getHeight(), 1);

        {
        ArrayHandle<int> h_handle(array_b, access_location::host, access_mode::read);
        UP_ASSERT(h_handle.data != NULL);
        for (int i = 0; i < (int)array_b.getNumElements(); i++)
            UP_ASSERT_EQUAL(h_handle.data[i], i);
        }

    // basic check 5: verify the = operator
    GPUArray<int> array_c(1, exec_conf);
    array_c = gpu_array;

    UP_ASSERT_EQUAL((int)array_c.getNumElements(), 100);
    UP_ASSERT_EQUAL((int)array_c.getPitch(), 100);
    UP_ASSERT_EQUAL((int)array_c.getHeight(), 1);

        {
        ArrayHandle<int> h_handle(array_c, access_location::host, access_mode::read);
        UP_ASSERT(h_handle.data != NULL);
        for (int i = 0; i < (int)array_c.getNumElements(); i++)
            UP_ASSERT_EQUAL(h_handle.data[i], i);
        }
    }

#ifdef ENABLE_HIP
//! test case for testing device to/from host transfers
UP_TEST(GPUArray_transfer_tests)
    {
    std::shared_ptr<ExecutionConfiguration> exec_conf(
        new ExecutionConfiguration(ExecutionConfiguration::GPU));
    UP_ASSERT(exec_conf->isCUDAEnabled());

    GPUArray<int> gpu_array(100, exec_conf);

        // initialize the data on the device
        {
        ArrayHandle<int> d_handle(gpu_array, access_location::device, access_mode::readwrite);
        UP_ASSERT(d_handle.data != NULL);

        gpu_fill_test_pattern(d_handle.data, gpu_array.getNumElements());
        hipError_t err_sync = hipPeekAtLastError();
        exec_conf->handleHIPError(err_sync, __FILE__, __LINE__);
        }

        // copy it to the host and verify
        {
        ArrayHandle<int> h_handle(gpu_array, access_location::host, access_mode::readwrite);
        UP_ASSERT(h_handle.data != NULL);
        for (int i = 0; i < (int)gpu_array.getNumElements(); i++)
            {
            UP_ASSERT_EQUAL(h_handle.data[i], i * i);
            // overwrite the data as we verify
            h_handle.data[i] = 100 + i;
            }
        }

        // data has been overwritten on the host. Increment it on the device in overwrite mode
        // and verify that the data was not copied from the host to device
        {
        ArrayHandle<int> d_handle(gpu_array, access_location::device, access_mode::overwrite);
        UP_ASSERT(d_handle.data != NULL);

        gpu_add_one(d_handle.data, gpu_array.getNumElements());
        hipError_t err_sync = hipPeekAtLastError();
        exec_conf->handleHIPError(err_sync, __FILE__, __LINE__);
        }

        // copy it back to the host and verify
        {
        ArrayHandle<int> h_handle(gpu_array, access_location::host, access_mode::readwrite);
        UP_ASSERT(h_handle.data != NULL);
        for (int i = 0; i < (int)gpu_array.getNumElements(); i++)
            {
            UP_ASSERT_EQUAL(h_handle.data[i], i * i + 1);
            // overwrite the data as we verify
            h_handle.data[i] = 100 + i;
            }
        }

        // access it on the device in read only mode, but be a bad boy and overwrite the data
        // the verify on the host should then still show the overwritten data as the internal state
        // should still be hostdevice and not copy the data back from the device
        {
        ArrayHandle<int> d_handle(gpu_array, access_location::device, access_mode::read);
        UP_ASSERT(d_handle.data != NULL);

        gpu_add_one(d_handle.data, gpu_array.getNumElements());
        hipError_t err_sync = hipPeekAtLastError();
        exec_conf->handleHIPError(err_sync, __FILE__, __LINE__);
        }

        {
        ArrayHandle<int> h_handle(gpu_array, access_location::host, access_mode::readwrite);
        UP_ASSERT(h_handle.data != NULL);
        for (int i = 0; i < (int)gpu_array.getNumElements(); i++)
            {
            UP_ASSERT_EQUAL(h_handle.data[i], 100 + i);
            }
        }

        // finally, test host-> device copies
        {
        ArrayHandle<int> d_handle(gpu_array, access_location::device, access_mode::readwrite);
        UP_ASSERT(d_handle.data != NULL);

        gpu_add_one(d_handle.data, gpu_array.getNumElements());
        hipError_t err_sync = hipPeekAtLastError();
        exec_conf->handleHIPError(err_sync, __FILE__, __LINE__);
        }

        // via the read access mode
        {
        ArrayHandle<int> h_handle(gpu_array, access_location::host, access_mode::read);
        UP_ASSERT(h_handle.data != NULL);
        for (int i = 0; i < (int)gpu_array.getNumElements(); i++)
            {
            UP_ASSERT_EQUAL(h_handle.data[i], 100 + i + 1);
            }
        }

        {
        ArrayHandle<int> d_handle(gpu_array, access_location::device, access_mode::readwrite);
        UP_ASSERT(d_handle.data != NULL);

        gpu_add_one(d_handle.data, gpu_array.getNumElements());
        hipError_t err_sync = hipPeekAtLastError();
        exec_conf->handleHIPError(err_sync, __FILE__, __LINE__);
        }

        // and via the readwrite access mode
        {
        ArrayHandle<int> h_handle(gpu_array, access_location::host, access_mode::readwrite);
        UP_ASSERT(h_handle.data != NULL);
        for (int i = 0; i < (int)gpu_array.getNumElements(); i++)
            {
            UP_ASSERT_EQUAL(h_handle.data[i], 100 + i + 1 + 1);
            }
        }
    }

//! Tests operations on NULL GPUArrays
UP_TEST(GPUArray_null_tests)
    {
    // Construct a NULL GPUArray
    GPUArray<int> a;

    UP_ASSERT(a.isNull());
    UP_ASSERT_EQUAL(a.getNumElements(), (unsigned)0);

    // check copy construction of a NULL GPUArray
    GPUArray<int> b(a);

    UP_ASSERT(b.isNull());
    UP_ASSERT_EQUAL(b.getNumElements(), (unsigned)0);

    // check assignment of a NULL GPUArray
    std::shared_ptr<ExecutionConfiguration> exec_conf(
        new ExecutionConfiguration(ExecutionConfiguration::GPU));
    GPUArray<int> c(1000, exec_conf);
    c = a;

    UP_ASSERT(c.isNull());
    UP_ASSERT_EQUAL(c.getNumElements(), (unsigned)0);

    // check swapping of a NULL GPUArray
    GPUArray<int> d(1000, exec_conf);

    d.swap(a);
    UP_ASSERT(d.isNull());
    UP_ASSERT_EQUAL(d.getNumElements(), (unsigned)0);

    UP_ASSERT(!a.isNull());
    UP_ASSERT_EQUAL(a.getNumElements(), (unsigned)1000);
    }

//! Tests resize methods
UP_TEST(GPUArray_resize_tests)
    {
    std::shared_ptr<ExecutionConfiguration> exec_conf(
        new ExecutionConfiguration(ExecutionConfiguration::GPU));

    // create a 1D GPUArray
    GPUArray<unsigned int> a(5, exec_conf);

        {
        // Fill it with some values
        ArrayHandle<unsigned int> h_handle(a, access_location::host, access_mode::overwrite);

        h_handle.data[0] = 1;
        h_handle.data[1] = 2;
        h_handle.data[2] = 3;
        h_handle.data[3] = 4;
        h_handle.data[4] = 5;
        }

    // resize the array
    a.resize(10);

    // check that it has the right size
    UP_ASSERT_EQUAL(a.getNumElements(), (unsigned int)10);
        {
        // test that it still contains the data
        ArrayHandle<unsigned int> h_handle(a, access_location::host, access_mode::read);

        UP_ASSERT_EQUAL(h_handle.data[0], (unsigned int)1);
        UP_ASSERT_EQUAL(h_handle.data[1], (unsigned int)2);
        UP_ASSERT_EQUAL(h_handle.data[2], (unsigned int)3);
        UP_ASSERT_EQUAL(h_handle.data[3], (unsigned int)4);
        UP_ASSERT_EQUAL(h_handle.data[4], (unsigned int)5);

        // test that the other elements are set to zero
        UP_ASSERT_EQUAL(h_handle.data[5], (unsigned int)0);
        UP_ASSERT_EQUAL(h_handle.data[6], (unsigned int)0);
        UP_ASSERT_EQUAL(h_handle.data[7], (unsigned int)0);
        UP_ASSERT_EQUAL(h_handle.data[8], (unsigned int)0);
        UP_ASSERT_EQUAL(h_handle.data[9], (unsigned int)0);
        }

    // check that it also works for a GPUArray that is initially empty
    GPUArray<unsigned int> b;
    b.resize(7);

    UP_ASSERT_EQUAL(b.getNumElements(), (unsigned int)7);

    // allocate a 2D GPUArray
    unsigned int width = 3;
    unsigned int height = 2;
    GPUArray<unsigned int> c(width, height, exec_conf);
    size_t pitch = c.getPitch();
        {
        // write some data to it
        ArrayHandle<unsigned int> h_handle(c, access_location::host, access_mode::overwrite);

        h_handle.data[0] = 123;
        h_handle.data[1] = 456;
        h_handle.data[2] = 789;

        h_handle.data[0 + pitch] = 1234;
        h_handle.data[1 + pitch] = 3456;
        h_handle.data[2 + pitch] = 5678;
        }

    // resize it
    width = 17;
    height = 4;
    c.resize(width, height);
    pitch = c.getPitch();

    // check that it has the right size
    UP_ASSERT_EQUAL(c.getNumElements(), pitch * height);

        {
        // test that we can still recover the data
        ArrayHandle<unsigned int> h_handle(c, access_location::host, access_mode::read);

        UP_ASSERT_EQUAL(h_handle.data[0], (unsigned int)123);
        UP_ASSERT_EQUAL(h_handle.data[1], (unsigned int)456);
        UP_ASSERT_EQUAL(h_handle.data[2], (unsigned int)789);

        // check that other elements of that row zero
        for (unsigned int i = 3; i < 17; i++)
            UP_ASSERT_EQUAL(h_handle.data[i], (unsigned int)0);

        UP_ASSERT_EQUAL(h_handle.data[0 + pitch], (unsigned int)1234);
        UP_ASSERT_EQUAL(h_handle.data[1 + pitch], (unsigned int)3456);
        UP_ASSERT_EQUAL(h_handle.data[2 + pitch], (unsigned int)5678);

        // check that other elements of that row are zero
        for (unsigned int i = 3; i < 17; i++)
            UP_ASSERT_EQUAL(h_handle.data[i + pitch], (unsigned int)0);

        // check that the two new rows are zero
        for (unsigned int i = 0; i < 17; i++)
            {
            UP_ASSERT_EQUAL(h_handle.data[i + 2 * pitch], (unsigned int)0);
            UP_ASSERT_EQUAL(h_handle.data[i + 3 * pitch], (unsigned int)0);
            }
        }
    }

//! Tests GPUVector
UP_TEST(GPUVector_basic_tests)
    {
    std::shared_ptr<ExecutionConfiguration> exec_conf(
        new ExecutionConfiguration(ExecutionConfiguration::GPU));

    // First create an empty GPUVector
    GPUVector<unsigned int> vec(exec_conf);

    // The size should be zero
    UP_ASSERT_EQUAL(vec.size(), (unsigned int)0);

    // Add some elements
    vec.push_back(1);
    vec.push_back(2);
    vec.push_back(3);

    UP_ASSERT_EQUAL(vec.size(), (unsigned int)3);

    // Test bracket operator
    UP_ASSERT_EQUAL((unsigned int)vec[0], (unsigned int)1);
    UP_ASSERT_EQUAL((unsigned int)vec[1], (unsigned int)2);
    UP_ASSERT_EQUAL((unsigned int)vec[2], (unsigned int)3);

    // Test assignment
    vec[1] = 4;

    UP_ASSERT_EQUAL((unsigned int)vec[0], (unsigned int)1);
    UP_ASSERT_EQUAL((unsigned int)vec[1], (unsigned int)4);
    UP_ASSERT_EQUAL((unsigned int)vec[2], (unsigned int)3);

    // remove an element
    vec.pop_back();

    UP_ASSERT_EQUAL(vec.size(), (unsigned int)2);

    // clear the array
    vec.clear();

    UP_ASSERT_EQUAL(vec.size(), (unsigned int)0);

    // resize it
    vec.resize(10);
    UP_ASSERT_EQUAL(vec.size(), (unsigned int)10);

    // verify we can still write to it
    vec[0] = 234;
    vec[1] = 123;
    vec[2] = 654;
    vec[3] = 789;
    vec[4] = 321;
    vec[5] = 432;
    vec[6] = 543;
    vec[7] = 678;
    vec[8] = 987;
    vec[9] = 890;

    // read back
    UP_ASSERT_EQUAL((unsigned int)vec[0], (unsigned int)234);
    UP_ASSERT_EQUAL((unsigned int)vec[1], (unsigned int)123);
    UP_ASSERT_EQUAL((unsigned int)vec[2], (unsigned int)654);
    UP_ASSERT_EQUAL((unsigned int)vec[3], (unsigned int)789);
    UP_ASSERT_EQUAL((unsigned int)vec[4], (unsigned int)321);
    UP_ASSERT_EQUAL((unsigned int)vec[5], (unsigned int)432);
    UP_ASSERT_EQUAL((unsigned int)vec[6], (unsigned int)543);
    UP_ASSERT_EQUAL((unsigned int)vec[7], (unsigned int)678);
    UP_ASSERT_EQUAL((unsigned int)vec[8], (unsigned int)987);
    UP_ASSERT_EQUAL((unsigned int)vec[9], (unsigned int)890);

    // test assignment operator
    GPUVector<unsigned int> v;
    v = vec;
    UP_ASSERT_EQUAL(v.size(), static_cast<unsigned int>(10));
    UP_ASSERT_EQUAL((unsigned int)v[0], (unsigned int)234);
    UP_ASSERT_EQUAL((unsigned int)v[1], (unsigned int)123);
    UP_ASSERT_EQUAL((unsigned int)v[2], (unsigned int)654);
    UP_ASSERT_EQUAL((unsigned int)v[3], (unsigned int)789);
    UP_ASSERT_EQUAL((unsigned int)v[4], (unsigned int)321);
    UP_ASSERT_EQUAL((unsigned int)v[5], (unsigned int)432);
    UP_ASSERT_EQUAL((unsigned int)v[6], (unsigned int)543);
    UP_ASSERT_EQUAL((unsigned int)v[7], (unsigned int)678);
    UP_ASSERT_EQUAL((unsigned int)v[8], (unsigned int)987);
    UP_ASSERT_EQUAL((unsigned int)v[9], (unsigned int)890);
    UP_ASSERT(static_cast<bool>(v.getExecutionConfiguration()));

    // test move assignment
    v = std::move(GPUVector<unsigned int>(
        5,
        2,
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU)));
    UP_ASSERT_EQUAL(v.size(), static_cast<unsigned int>(5));
    UP_ASSERT_EQUAL((unsigned int)v[0], (unsigned int)2);
    UP_ASSERT_EQUAL((unsigned int)v[1], (unsigned int)2);
    UP_ASSERT_EQUAL((unsigned int)v[2], (unsigned int)2);
    UP_ASSERT_EQUAL((unsigned int)v[3], (unsigned int)2);
    UP_ASSERT_EQUAL((unsigned int)v[4], (unsigned int)2);

    // test move constructor
    GPUVector<unsigned int> v2 = std::move(GPUVector<unsigned int>(
        4,
        3,
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU)));
    UP_ASSERT_EQUAL(v2.size(), static_cast<unsigned int>(4));
    UP_ASSERT_EQUAL((unsigned int)v2[0], (unsigned int)3);
    UP_ASSERT_EQUAL((unsigned int)v2[1], (unsigned int)3);
    UP_ASSERT_EQUAL((unsigned int)v2[2], (unsigned int)3);
    UP_ASSERT_EQUAL((unsigned int)v2[3], (unsigned int)3);
    }
#endif

#pragma GCC diagnostic pop
