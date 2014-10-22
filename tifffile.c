/* tifffile.c

 A Python C extension module for decoding PackBits and LZW encoded TIFF data.

 Refer to the tifffile.py module for documentation and tests.

 :Author:
 `Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/>`_

 :Organization:
 Laboratory for Fluorescence Dynamics, University of California, Irvine

 :Version: 2013.11.05

 Install
 -------
 Use this Python distutils setup script to build the extension module::

 # setup.py
 # Usage: ``python setup.py build_ext --inplace``
 from distutils.core import setup, Extension
 import numpy
 setup(name='_tifffile',
 ext_modules=[Extension('_tifffile', ['tifffile.c'],
 include_dirs=[numpy.get_include()])])

 License
 -------
 Copyright (c) 2008-2014, Christoph Gohlke
 Copyright (c) 2008-2014, The Regents of the University of California
 Produced at the Laboratory for Fluorescence Dynamics
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions and the following disclaimer in the
 documentation and/or other materials provided with the distribution.
 * Neither the name of the copyright holders nor the names of any
 contributors may be used to endorse or promote products derived
 from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.
 */

#define _VERSION_ "2013.11.05"

#define WIN32_LEAN_AND_MEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"
#include "string.h"
#include "numpy/arrayobject.h"

/* little endian by default */
#ifndef MSB
#define MSB 1
#endif

#if MSB
#define LSB 0
#define BOC '<'
#else
#define LSB 1
#define BOC '>'
#endif

#define NO_ERROR 0
#define VALUE_ERROR -1

#ifdef _MSC_VER
typedef unsigned __int8  uint8_t;
typedef unsigned __int16  uint16_t;
typedef unsigned __int32  uint32_t;
typedef unsigned __int64  uint64_t;
#ifdef _WIN64
typedef __int64  ssize_t;
typedef signed __int64  intptr_t;
typedef unsigned __int64  uintptr_t;
#define SSIZE_MAX (9223372036854775808)
#else
typedef int ssize_t;
typedef _W64 signed int  intptr_t;
typedef _W64 unsigned int  uintptr_t;
#define SSIZE_MAX (2147483648)
#endif
#else
/* non MS compilers */
#include <stdint.h>
#include <limits.h>
#endif

#define SWAP2BYTES(x) \
((((x) >> 8) & 0x00FF) | (((x) & 0x00FF) << 8))

#define SWAP4BYTES(x) \
((((x) >> 24) & 0x00FF) | (((x)&0x00FF) << 24) | \
(((x) >> 8 ) & 0xFF00) | (((x)&0xFF00) << 8))

#define SWAP8BYTES(x) \
((((x) >> 56) & 0x00000000000000FF) | (((x) >> 40) & 0x000000000000FF00) | \
(((x) >> 24) & 0x0000000000FF0000) | (((x) >> 8)  & 0x00000000FF000000) | \
(((x) << 8)  & 0x000000FF00000000) | (((x) << 24) & 0x0000FF0000000000) | \
(((x) << 40) & 0x00FF000000000000) | (((x) << 56) & 0xFF00000000000000))

struct BYTE_STRING {
    unsigned int ref; /* reference count */
    unsigned int len; /* length of string */
    char *str;        /* pointer to bytes */
};

typedef union {
    uint8_t b[2];
    uint16_t i;
} u_uint16;

typedef union {
    uint8_t b[4];
    uint32_t i;
} u_uint32;

typedef union {
    uint8_t b[8];
    uint64_t i;
} u_uint64;

/*****************************************************************************/
/* C functions */

/* Return mask for itemsize bits */
unsigned char bitmask(const int itemsize) {
    unsigned char result = 0;
    unsigned char power = 1;
    int i;
    for (i = 0; i < itemsize; i++) {
        result += power;
        power *= 2;
    }
    return result << (8 - itemsize);
}

/** Unpack sequence of tigthly packed 1-32 bit integers.

 Native byte order will be returned.

 Input data array should be padded to the next 16, 32 or 64-bit boundary
 if itemsize not in (1, 2, 4, 8, 16, 24, 32, 64).

 */
int unpackbits(
               unsigned char *data,
               const ssize_t size,  /** size of data in bytes */
               const int itemsize,  /** number of bits in integer */
               ssize_t numitems,  /** number of items to unpack */
               unsigned char *result  /** buffer to store unpacked items */
               )
{
    ssize_t i, j, k, storagesize;
    unsigned char value;
    /* Input validation is done in wrapper function */
    storagesize = (ssize_t)(ceil(itemsize / 8.0));
    storagesize = storagesize < 3 ? storagesize : storagesize > 4 ? 8 : 4;
    switch (itemsize) {
        case 8:
        case 16:
        case 32:
        case 64:
            memcpy(result, data, numitems*storagesize);
            return NO_ERROR;
        case 1:
            for (i = 0, j = 0; i < numitems/8; i++) {
                value = data[i];
                result[j++] = (value & (unsigned char)(128)) >> 7;
                result[j++] = (value & (unsigned char)(64)) >> 6;
                result[j++] = (value & (unsigned char)(32)) >> 5;
                result[j++] = (value & (unsigned char)(16)) >> 4;
                result[j++] = (value & (unsigned char)(8)) >> 3;
                result[j++] = (value & (unsigned char)(4)) >> 2;
                result[j++] = (value & (unsigned char)(2)) >> 1;
                result[j++] = (value & (unsigned char)(1));
            }
            if (numitems % 8) {
                value = data[i];
                switch (numitems % 8) {
                    case 7: result[j+6] = (value & (unsigned char)(2)) >> 1;
                    case 6: result[j+5] = (value & (unsigned char)(4)) >> 2;
                    case 5: result[j+4] = (value & (unsigned char)(8)) >> 3;
                    case 4: result[j+3] = (value & (unsigned char)(16)) >> 4;
                    case 3: result[j+2] = (value & (unsigned char)(32)) >> 5;
                    case 2: result[j+1] = (value & (unsigned char)(64)) >> 6;
                    case 1: result[j] = (value & (unsigned char)(128)) >> 7;
                }
            }
            return NO_ERROR;
        case 2:
            for (i = 0, j = 0; i < numitems/4; i++) {
                value = data[i];
                result[j++] = (value & (unsigned char)(192)) >> 6;
                result[j++] = (value & (unsigned char)(48)) >> 4;
                result[j++] = (value & (unsigned char)(12)) >> 2;
                result[j++] = (value & (unsigned char)(3));
            }
            if (numitems % 4) {
                value = data[i];
                switch (numitems % 4) {
                    case 3: result[j+2] = (value & (unsigned char)(12)) >> 2;
                    case 2: result[j+1] = (value & (unsigned char)(48)) >> 4;
                    case 1: result[j] = (value & (unsigned char)(192)) >> 6;
                }
            }
            return NO_ERROR;
        case 4:
            for (i = 0, j = 0; i < numitems/2; i++) {
                value = data[i];
                result[j++] = (value & (unsigned char)(240)) >> 4;
                result[j++] = (value & (unsigned char)(15));
            }
            if (numitems % 2) {
                value = data[i];
                result[j] = (value & (unsigned char)(240)) >> 4;
            }
            return NO_ERROR;
        case 24:
            j = k = 0;
            for (i = 0; i < numitems; i++) {
                result[j++] = 0;
                result[j++] = data[k++];
                result[j++] = data[k++];
                result[j++] = data[k++];
            }
            return NO_ERROR;
    }
    /* 3, 5, 6, 7 */
    if (itemsize < 8) {
        int shr = 16;
        u_uint16 value, mask, tmp;
        j = k = 0;
        value.b[MSB] = data[j++];
        value.b[LSB] = data[j++];
        mask.b[MSB] = bitmask(itemsize);
        mask.b[LSB] = 0;
        for (i = 0; i < numitems; i++) {
            shr -= itemsize;
            tmp.i = (value.i & mask.i) >> shr;
            result[k++] = tmp.b[LSB];
            if (shr < itemsize) {
                value.b[MSB] = value.b[LSB];
                value.b[LSB] = data[j++];
                mask.i <<= 8 - itemsize;
                shr += 8;
            } else {
                mask.i >>= itemsize;
            }
        }
        return NO_ERROR;
    }
    /* 9, 10, 11, 12, 13, 14, 15 */
    if (itemsize < 16) {
        int shr = 32;
        u_uint32 value, mask, tmp;
        mask.i = 0;
        j = k = 0;
#if MSB
        for (i = 3; i >= 0; i--) {
            value.b[i] = data[j++];
        }
        mask.b[3] = 0xFF;
        mask.b[2] = bitmask(itemsize-8);
        for (i = 0; i < numitems; i++) {
            shr -= itemsize;
            tmp.i = (value.i & mask.i) >> shr;
            result[k++] = tmp.b[0]; /* swap bytes */
            result[k++] = tmp.b[1];
            if (shr < itemsize) {
                value.b[3] = value.b[1];
                value.b[2] = value.b[0];
                value.b[1] = data[j++];
                value.b[0] = data[j++];
                mask.i <<= 16 - itemsize;
                shr += 16;
            } else {
                mask.i >>= itemsize;
            }
        }
#else
        /* not implemented */
#endif
        return NO_ERROR;
    }
    /* 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31 */
    if (itemsize < 32) {
        int shr = 64;
        u_uint64 value, mask, tmp;
        mask.i = 0;
        j = k = 0;
#if MSB
        for (i = 7; i >= 0; i--) {
            value.b[i] = data[j++];
        }
        mask.b[7] = 0xFF;
        mask.b[6] = 0xFF;
        mask.b[5] = itemsize > 23 ? 0xFF : bitmask(itemsize-16);
        mask.b[4] = itemsize < 24 ? 0x00 : bitmask(itemsize-24);
        for (i = 0; i < numitems; i++) {
            shr -= itemsize;
            tmp.i = (value.i & mask.i) >> shr;
            result[k++] = tmp.b[0]; /* swap bytes */
            result[k++] = tmp.b[1];
            result[k++] = tmp.b[2];
            result[k++] = tmp.b[3];
            if (shr < itemsize) {
                value.b[7] = value.b[3];
                value.b[6] = value.b[2];
                value.b[5] = value.b[1];
                value.b[4] = value.b[0];
                value.b[3] = data[j++];
                value.b[2] = data[j++];
                value.b[1] = data[j++];
                value.b[0] = data[j++];
                mask.i <<= 32 - itemsize;
                shr += 32;
            } else {
                mask.i >>= itemsize;
            }
        }
#else
        /* Not implemented */
#endif
        return NO_ERROR;
    }
    return VALUE_ERROR;
}

/*****************************************************************************/
/* Python functions */

/** Unpack tightly packed integers. */
char py_unpackints_doc[] = "Unpack groups of bits into numpy array.";

static PyObject*
py_unpackints(PyObject *obj, PyObject *args, PyObject *kwds)
{
    PyObject *byteobj = NULL;
    PyArrayObject *result = NULL;
    PyArray_Descr *dtype = NULL;
    char *encoded = NULL;
    char *decoded = NULL;
    Py_ssize_t encoded_len = 0;
    Py_ssize_t decoded_len = 0;
    Py_ssize_t runlen = 0;
    Py_ssize_t i;
    int storagesize, bytesize;
    int itemsize = 0;
    int skipbits = 0;
    static char *kwlist[] = {"data", "dtype", "itemsize", "runlen", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO&i|i", kwlist,
                                     &byteobj, PyArray_DescrConverter, &dtype, &itemsize, &runlen))
        return NULL;

    Py_INCREF(byteobj);

    if (((itemsize < 1) || (itemsize > 32)) && (itemsize != 64)) {
        PyErr_Format(PyExc_ValueError, "itemsize out of range");
        goto _fail;
    }

    if (!PyBytes_Check(byteobj)) {
        PyErr_Format(PyExc_TypeError, "expected byte string as input");
        goto _fail;
    }

    encoded = PyBytes_AS_STRING(byteobj);
    encoded_len = PyBytes_GET_SIZE(byteobj);
    bytesize = (int)ceil(itemsize / 8.0);
    storagesize = bytesize < 3 ? bytesize : bytesize > 4 ? 8 : 4;
    if ((encoded_len < bytesize) || (encoded_len > SSIZE_MAX / storagesize)) {
        PyErr_Format(PyExc_ValueError, "data size out of range");
        goto _fail;
    }
    if (dtype->elsize != storagesize) {
        PyErr_Format(PyExc_TypeError, "dtype.elsize doesn't fit itemsize");
        goto _fail;
    }

    if (runlen == 0) {
        runlen = (Py_ssize_t)(((uint64_t)encoded_len*8) / (uint64_t)itemsize);
    }
    skipbits = (Py_ssize_t)(((uint64_t)runlen * (uint64_t)itemsize) % 8);
    if (skipbits > 0) {
        skipbits = 8 - skipbits;
    }
    decoded_len = (Py_ssize_t)((uint64_t)runlen * (((uint64_t)encoded_len*8) /
                                                   ((uint64_t)runlen*(uint64_t)itemsize + (uint64_t)skipbits)));

    result = (PyArrayObject *)PyArray_SimpleNew(1, &decoded_len,
                                                dtype->type_num);
    if (result == NULL) {
        PyErr_Format(PyExc_MemoryError, "unable to allocate output array");
        goto _fail;
    }
    decoded = (char *)PyArray_DATA(result);

    for (i = 0; i < decoded_len; i+=runlen) {
        if (NO_ERROR !=
            unpackbits((unsigned char *) encoded,
                       (ssize_t) encoded_len,
                       (int) itemsize,
                       (ssize_t) runlen,
                       (unsigned char *) decoded)) {
                PyErr_Format(PyExc_ValueError, "unpackbits() failed");
                goto _fail;
            }
        encoded += (Py_ssize_t)(((uint64_t)runlen * (uint64_t)itemsize +
                                 (uint64_t)skipbits) / 8);
        decoded += runlen * storagesize;
    }

    if ((dtype->byteorder != BOC) && (itemsize % 8 == 0)) {
        switch (dtype->elsize) {
            case 2: {
                uint16_t *d = (uint16_t *)PyArray_DATA(result);
                for (i = 0; i < PyArray_SIZE(result); i++) {
                    *d = SWAP2BYTES(*d); d++;
                }
                break; }
            case 4: {
                uint32_t *d = (uint32_t *)PyArray_DATA(result);
                for (i = 0; i < PyArray_SIZE(result); i++) {
                    *d = SWAP4BYTES(*d); d++;
                }
                break; }
            case 8: {
                uint64_t *d = (uint64_t *)PyArray_DATA(result);
                for (i = 0; i < PyArray_SIZE(result); i++) {
                    *d = SWAP8BYTES(*d); d++;
                }
                break; }
        }
    }
    Py_DECREF(byteobj);
    Py_DECREF(dtype);
    return PyArray_Return(result);

_fail:
    Py_XDECREF(byteobj);
    Py_XDECREF(result);
    Py_XDECREF(dtype);
    return NULL;
}


/** Decode TIFF PackBits encoded string. */
char py_decodepackbits_doc[] = "Return TIFF PackBits decoded string.";

static PyObject *
py_decodepackbits(PyObject *obj, PyObject *args)
{
    int n;
    char e;
    char *decoded = NULL;
    char *encoded = NULL;
    char *encoded_end = NULL;
    char *encoded_pos = NULL;
    unsigned int encoded_len;
    unsigned int decoded_len;
    PyObject *byteobj = NULL;
    PyObject *result = NULL;

    if (!PyArg_ParseTuple(args, "O", &byteobj))
        return NULL;

    if (!PyBytes_Check(byteobj)) {
        PyErr_Format(PyExc_TypeError, "expected byte string as input");
        goto _fail;
    }

    Py_INCREF(byteobj);
    encoded = PyBytes_AS_STRING(byteobj);
    encoded_len = (unsigned int)PyBytes_GET_SIZE(byteobj);

    /* release GIL: byte/string objects are immutable */
    Py_BEGIN_ALLOW_THREADS

    /* determine size of decoded string */
    encoded_pos = encoded;
    encoded_end = encoded + encoded_len;
    decoded_len = 0;
    while (encoded_pos < encoded_end) {
        n = (int)*encoded_pos++;
        if (n >= 0) {
            n++;
            if (encoded_pos+n > encoded_end)
                n = (int)(encoded_end - encoded_pos);
            encoded_pos += n;
            decoded_len += n;
        } else if (n > -128) {
            encoded_pos++;
            decoded_len += 1-n;
        }
    }
    Py_END_ALLOW_THREADS

    result = PyBytes_FromStringAndSize(0, decoded_len);
    if (result == NULL) {
        PyErr_Format(PyExc_MemoryError, "failed to allocate decoded string");
        goto _fail;
    }
    decoded = PyBytes_AS_STRING(result);

    Py_BEGIN_ALLOW_THREADS

    /* decode string */
    encoded_end = encoded + encoded_len;
    while (encoded < encoded_end) {
        n = (int)*encoded++;
        if (n >= 0) {
            n++;
            if (encoded+n > encoded_end)
                n = (int)(encoded_end - encoded);
            /* memmove(decoded, encoded, n); decoded += n; encoded += n; */
            while (n--)
                *decoded++ = *encoded++;
        } else if (n > -128) {
            n = 1 - n;
            e = *encoded++;
            /* memset(decoded, e, n); decoded += n; */
            while (n--)
                *decoded++ = e;
        }
    }
    Py_END_ALLOW_THREADS
    
    Py_DECREF(byteobj);
    return result;
    
_fail:
    Py_XDECREF(byteobj);
    Py_XDECREF(result);
    return NULL;
}


/** Decode TIFF LZW encoded string. */
char py_decodelzw_doc[] = "Return TIFF LZW decoded string.";

static PyObject *
py_decodelzw(PyObject *obj, PyObject *args)
{
    PyThreadState *_save = NULL;