# TinyEst - Tiny Estimator

Daniël de Kok &lt;<me@danieldk.eu>&gt;

## Introduction

TinyEst (pronounced *tiniest*) is a maximum entropy model parameter
optimizer. It uses the liblbfgs library to minimize the negative
log-likelihood of the model. The program emphasizes simplicity and clarity
of data structures.

## Compilation

Compilation of TinyEst requires a C99 compiler and the CMake build tool,
and can be performed in two simple steps:

    cmake .
    make

Afterwards you will find the 'tinyest' library and 'estimate' utility.

## License

    Copyright 2010 Daniël de Kok
    
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
  
        http://www.apache.org/licenses/LICENSE-2.0
  
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

TinyEst includes the liblbfgs library, to which the following license
notice applies:


    Copyright (c) 1990, Jorge Nocedal
    Copyright (c) 2007-2010 Naoaki Okazaki
    All rights reserved.
    
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.
