// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#pragma once
#include "GPUArray.h"
#include <algorithm>

// The factor with which the array size is incremented
#define RESIZE_FACTOR 9.f / 8.f

namespace hoomd
    {
//! Forward declarations
template<class T> class GPUVector;

//! Class for managing a vector of elements on the GPU mirrored to the CPU
/*! The GPUVector class is a simple container for a variable number of elements. Its interface
   is inspired by std::vector and it offers methods to insert new elements at the end of the list,
   and to remove them from there.

    It uses a GPUArray as the underlying storage class, thus the data in a GPUVector can also be
   accessed directly using ArrayHandles.

    In the current implementation, a GPUVector can only grow (but not shrink) in size until it
   is destroyed.

    \ingroup data_structs
*/
template<class T> class GPUVector : public GPUArray<T>
    {
    public:
    //! Default constructor
    GPUVector();

    //! Constructs an empty GPUVector
    GPUVector(std::shared_ptr<const ExecutionConfiguration> exec_conf);

    //! Constructs a GPUVector
    GPUVector(size_t size, std::shared_ptr<const ExecutionConfiguration> exec_conf);

    //! Constructs a GPUVector of given size, initialized with a constant value
    GPUVector(unsigned int size,
              const T& value,
              std::shared_ptr<const ExecutionConfiguration> exec_conf);

#ifdef ENABLE_HIP
    //! Constructs an empty GPUVector
    GPUVector(std::shared_ptr<const ExecutionConfiguration> exec_conf, bool mapped);

    //! Constructs a GPUVector
    GPUVector(size_t size, std::shared_ptr<const ExecutionConfiguration> exec_conf, bool mapped);
#endif
    //! Frees memory
    ~GPUVector() = default;

    //! Copy constructor
    GPUVector(const GPUVector& from);
    //! Move constructor
    GPUVector(GPUVector&& other);
    //! = operator
    GPUVector& operator=(const GPUVector& rhs);
    //! Move assignment operator
    GPUVector& operator=(GPUVector&& other);

    //! swap this GPUVectorwith another
    inline void swap(GPUVector& from);

    /*!
      \returns the current size of the vector
    */
    size_t size() const
        {
        return m_size;
        }

    //! Resize the GPUVector
    /*! \param new_size New number of elements
     */
    void resize(size_t new_size);

    //! Resize the GPUVector
    /*! \param new_size New number of elements
     *  \param const value to initialize newly added elements with
     */
    void resize(size_t new_size, const T& value);

    //! Insert an element at the end of the vector
    /*! \param val The new element
     */
    void push_back(const T& val);

    //! Remove an element from the end of the list
    void pop_back();

    //! Remove an element by index
    void erase(size_t i);

    //! Clear the list
    void clear();

    //! Proxy class to provide access to the data elements of the vector
    class data_proxy
        {
        public:
        //! Constructor
        data_proxy(const GPUVector<T>& _vec, const size_t _n) : vec(_vec), n(_n) { }

        //! Type cast
        operator T() const
            {
            T* data = vec.acquireHost(access_mode::read);
            T val = data[n];
            vec.release();
            return val;
            }

        //! Assignment
        data_proxy& operator=(T rhs)
            {
            T* data = vec.acquireHost(access_mode::readwrite);
            data[n] = rhs;
            vec.release();
            return *this;
            }

        private:
        const GPUVector<T>& vec; //!< The vector that is accessed
        size_t n;                //!< The index of the element to access
        };

    //! Get a proxy-reference to a list element
    data_proxy operator[](size_t n)
        {
        assert(n < m_size);
        return data_proxy(*this, n);
        }

    //! Get a proxy-reference to a list element (const version)
    data_proxy operator[](size_t n) const
        {
        assert(n < m_size);
        return data_proxy(*this, n);
        }

    private:
    size_t m_size; //!< Number of elements

    //! Helper function to reallocate the GPUArray (using amortized array resizing)
    void reallocate(size_t new_size);

    //! Acquire the underlying GPU array on the host
    T* acquireHost(const access_mode::Enum mode) const;

    friend class data_proxy;
    };

//******************************************
// GPUVector implementation
// *****************************************

//! Default constructor
/*! \warning When using this constructor, a properly initialized GPUVector with an exec_conf
   needs to be swapped in later, after construction of the GPUVector.
 */
template<class T> GPUVector<T>::GPUVector() : m_size(0) { }

/*! \param exec_conf Shared pointer to the execution configuration
 */
template<class T>
GPUVector<T>::GPUVector(std::shared_ptr<const ExecutionConfiguration> exec_conf)
    : GPUArray<T>(0, exec_conf), m_size(0)
    {
    }

/*! \param size Number of elements to allocate initial memory for in the array
    \param exec_conf Shared pointer to the execution configuration
*/
template<class T>
GPUVector<T>::GPUVector(size_t size, std::shared_ptr<const ExecutionConfiguration> exec_conf)
    : GPUArray<T>(size, exec_conf), m_size(size)
    {
    }

/*! \param size Number of elements to allocate initial memory for in the array
    \param value constant value to initialize the array with
    \param exec_conf Shared pointer to the execution configuration
*/
template<class T>
GPUVector<T>::GPUVector(unsigned int size,
                        const T& value,
                        std::shared_ptr<const ExecutionConfiguration> exec_conf)
    : GPUArray<T>(size, exec_conf), m_size(size)
    {
    T* data = acquireHost(access_mode::readwrite);
    for (unsigned int i = 0; i < size; ++i)
        data[i] = value;
    this->release();
    }

template<class T>
GPUVector<T>::GPUVector(const GPUVector& from) : GPUArray<T>(from), m_size(from.m_size)
    {
    }

template<class T>
GPUVector<T>::GPUVector(GPUVector&& other)
    : GPUArray<T>(std::move(other)), m_size(std::move(other.m_size))
    {
    }

template<class T> GPUVector<T>& GPUVector<T>::operator=(const GPUVector& rhs)
    {
    if (this != &rhs) // protect against invalid self-assignment
        {
        m_size = rhs.m_size;
        // invoke base class operator
        GPUArray<T>::operator=(rhs);
        }

    return *this;
    }

template<class T> GPUVector<T>& GPUVector<T>::operator=(GPUVector&& other)
    {
    if (this != &other)
        {
        m_size = std::move(other.m_size);
        // invoke move assignment for the base class
        GPUArray<T>::operator=(std::move(other));
        }

    return *this;
    }

/*! \param from GPUVector to swap \a this with
 */
template<class T> void GPUVector<T>::swap(GPUVector<T>& from)
    {
    std::swap(m_size, from.m_size);
    GPUArray<T>::swap(from);
    }

/*! \param size New requested size of allocated memory
 *
 * Internally, this method uses amortized resizing of allocated memory to
 * avoid excessive copying of data. The GPUArray is only reallocated if necessary,
 * i.e. if the requested size is larger than the current size, which is a power of two.
 */
template<class T> void GPUVector<T>::reallocate(size_t size)
    {
    if (size > GPUArray<T>::getNumElements())
        {
        // reallocate
        size_t new_allocated_size
            = GPUArray<T>::getNumElements() ? GPUArray<T>::getNumElements() : 1;

        // double the size as often as necessary
        while (size > new_allocated_size)
            new_allocated_size = ((size_t)(((double)new_allocated_size) * RESIZE_FACTOR)) + 1;

        // actually resize the underlying GPUArray
        GPUArray<T>::resize(new_allocated_size);
        }
    }

/*! \param new_size New size of vector
 \post The GPUVector will be re-allocated if necessary to hold the new elements.
       The newly allocated memory is \b not initialized. It is responsibility of the caller to
 ensure correct initialization, e.g. using clear()
*/
template<class T> void GPUVector<T>::resize(size_t new_size)
    {
    // allocate memory by amortized O(N) resizing
    if (new_size > 0)
        reallocate(new_size);
    else
        // for zero size, we at least allocate the memory
        reallocate(1);

    // set new size
    m_size = new_size;
    }

/*!
 \post The GPUVector will be re-allocated if necessary to hold the new elements.
       The newly allocated memory is initialized with a constant value.
*/
template<class T> void GPUVector<T>::resize(size_t new_size, const T& value)
    {
    size_t old_size = m_size;
    resize(new_size);
    T* data = acquireHost(access_mode::readwrite);
    for (size_t i = old_size; i < new_size; ++i)
        data[i] = value;
    this->release();
    }

//! Insert an element at the end of the vector
template<class T> void GPUVector<T>::push_back(const T& val)
    {
    reallocate(m_size + 1);

    T* data = acquireHost(access_mode::readwrite);
    data[m_size++] = val;
    this->release();
    }

//! Remove an element from the end of the list
template<class T> void GPUVector<T>::pop_back()
    {
    assert(m_size);
    m_size--;
    }

//! Remove an element in the middle
template<class T> void GPUVector<T>::erase(size_t i)
    {
    assert(i < m_size);
    T* data = acquireHost(access_mode::readwrite);

    T* res = data;
    for (size_t n = 0; n < m_size; ++n)
        {
        if (n != i)
            {
            *res = *data;
            res++;
            }
        data++;
        }
    m_size--;
    this->release();
    }

//! Clear the list
template<class T> void GPUVector<T>::clear()
    {
    m_size = 0;
    }

/*! \param mode Access mode for the GPUArray
 */
template<class T> T* GPUVector<T>::acquireHost(const access_mode::Enum mode) const
    {
#ifdef ENABLE_HIP
    return GPUArray<T>::acquire(access_location::host, access_mode::readwrite, false);
#else
    return GPUArray<T>::acquire(access_location::host, access_mode::readwrite);
#endif
    }

    } // end namespace hoomd
