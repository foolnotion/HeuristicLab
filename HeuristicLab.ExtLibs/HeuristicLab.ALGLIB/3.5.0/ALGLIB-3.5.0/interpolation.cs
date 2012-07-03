/*************************************************************************
Copyright (c) Sergey Bochkanov (ALGLIB project).

>>> SOURCE LICENSE >>>
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation (www.fsf.org); either version 2 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

A copy of the GNU General Public License is available at
http://www.fsf.org/licensing/licenses
>>> END OF LICENSE >>>
*************************************************************************/
#pragma warning disable 162
#pragma warning disable 219
using System;

public partial class alglib
{


    /*************************************************************************
    IDW interpolant.
    *************************************************************************/
    public class idwinterpolant
    {
        //
        // Public declarations
        //

        public idwinterpolant()
        {
            _innerobj = new idwint.idwinterpolant();
        }

        //
        // Although some of declarations below are public, you should not use them
        // They are intended for internal use only
        //
        private idwint.idwinterpolant _innerobj;
        public idwint.idwinterpolant innerobj { get { return _innerobj; } }
        public idwinterpolant(idwint.idwinterpolant obj)
        {
            _innerobj = obj;
        }
    }

    /*************************************************************************
    IDW interpolation

    INPUT PARAMETERS:
        Z   -   IDW interpolant built with one of model building
                subroutines.
        X   -   array[0..NX-1], interpolation point

    Result:
        IDW interpolant Z(X)

      -- ALGLIB --
         Copyright 02.03.2010 by Bochkanov Sergey
    *************************************************************************/
    public static double idwcalc(idwinterpolant z, double[] x)
    {

        double result = idwint.idwcalc(z.innerobj, x);
        return result;
    }

    /*************************************************************************
    IDW interpolant using modified Shepard method for uniform point
    distributions.

    INPUT PARAMETERS:
        XY  -   X and Y values, array[0..N-1,0..NX].
                First NX columns contain X-values, last column contain
                Y-values.
        N   -   number of nodes, N>0.
        NX  -   space dimension, NX>=1.
        D   -   nodal function type, either:
                * 0     constant  model.  Just  for  demonstration only, worst
                        model ever.
                * 1     linear model, least squares fitting. Simpe  model  for
                        datasets too small for quadratic models
                * 2     quadratic  model,  least  squares  fitting. Best model
                        available (if your dataset is large enough).
                * -1    "fast"  linear  model,  use  with  caution!!!   It  is
                        significantly  faster than linear/quadratic and better
                        than constant model. But it is less robust (especially
                        in the presence of noise).
        NQ  -   number of points used to calculate  nodal  functions  (ignored
                for constant models). NQ should be LARGER than:
                * max(1.5*(1+NX),2^NX+1) for linear model,
                * max(3/4*(NX+2)*(NX+1),2^NX+1) for quadratic model.
                Values less than this threshold will be silently increased.
        NW  -   number of points used to calculate weights and to interpolate.
                Required: >=2^NX+1, values less than this  threshold  will  be
                silently increased.
                Recommended value: about 2*NQ

    OUTPUT PARAMETERS:
        Z   -   IDW interpolant.

    NOTES:
      * best results are obtained with quadratic models, worst - with constant
        models
      * when N is large, NQ and NW must be significantly smaller than  N  both
        to obtain optimal performance and to obtain optimal accuracy. In 2  or
        3-dimensional tasks NQ=15 and NW=25 are good values to start with.
      * NQ  and  NW  may  be  greater  than  N.  In  such  cases  they will be
        automatically decreased.
      * this subroutine is always succeeds (as long as correct parameters  are
        passed).
      * see  'Multivariate  Interpolation  of Large Sets of Scattered Data' by
        Robert J. Renka for more information on this algorithm.
      * this subroutine assumes that point distribution is uniform at the small
        scales.  If  it  isn't  -  for  example,  points are concentrated along
        "lines", but "lines" distribution is uniform at the larger scale - then
        you should use IDWBuildModifiedShepardR()


      -- ALGLIB PROJECT --
         Copyright 02.03.2010 by Bochkanov Sergey
    *************************************************************************/
    public static void idwbuildmodifiedshepard(double[,] xy, int n, int nx, int d, int nq, int nw, out idwinterpolant z)
    {
        z = new idwinterpolant();
        idwint.idwbuildmodifiedshepard(xy, n, nx, d, nq, nw, z.innerobj);
        return;
    }

    /*************************************************************************
    IDW interpolant using modified Shepard method for non-uniform datasets.

    This type of model uses  constant  nodal  functions and interpolates using
    all nodes which are closer than user-specified radius R. It  may  be  used
    when points distribution is non-uniform at the small scale, but it  is  at
    the distances as large as R.

    INPUT PARAMETERS:
        XY  -   X and Y values, array[0..N-1,0..NX].
                First NX columns contain X-values, last column contain
                Y-values.
        N   -   number of nodes, N>0.
        NX  -   space dimension, NX>=1.
        R   -   radius, R>0

    OUTPUT PARAMETERS:
        Z   -   IDW interpolant.

    NOTES:
    * if there is less than IDWKMin points within  R-ball,  algorithm  selects
      IDWKMin closest ones, so that continuity properties of  interpolant  are
      preserved even far from points.

      -- ALGLIB PROJECT --
         Copyright 11.04.2010 by Bochkanov Sergey
    *************************************************************************/
    public static void idwbuildmodifiedshepardr(double[,] xy, int n, int nx, double r, out idwinterpolant z)
    {
        z = new idwinterpolant();
        idwint.idwbuildmodifiedshepardr(xy, n, nx, r, z.innerobj);
        return;
    }

    /*************************************************************************
    IDW model for noisy data.

    This subroutine may be used to handle noisy data, i.e. data with noise  in
    OUTPUT values.  It differs from IDWBuildModifiedShepard() in the following
    aspects:
    * nodal functions are not constrained to pass through  nodes:  Qi(xi)<>yi,
      i.e. we have fitting  instead  of  interpolation.
    * weights which are used during least  squares fitting stage are all equal
      to 1.0 (independently of distance)
    * "fast"-linear or constant nodal functions are not supported (either  not
      robust enough or too rigid)

    This problem require far more complex tuning than interpolation  problems.
    Below you can find some recommendations regarding this problem:
    * focus on tuning NQ; it controls noise reduction. As for NW, you can just
      make it equal to 2*NQ.
    * you can use cross-validation to determine optimal NQ.
    * optimal NQ is a result of complex tradeoff  between  noise  level  (more
      noise = larger NQ required) and underlying  function  complexity  (given
      fixed N, larger NQ means smoothing of compex features in the data).  For
      example, NQ=N will reduce noise to the minimum level possible,  but  you
      will end up with just constant/linear/quadratic (depending on  D)  least
      squares model for the whole dataset.

    INPUT PARAMETERS:
        XY  -   X and Y values, array[0..N-1,0..NX].
                First NX columns contain X-values, last column contain
                Y-values.
        N   -   number of nodes, N>0.
        NX  -   space dimension, NX>=1.
        D   -   nodal function degree, either:
                * 1     linear model, least squares fitting. Simpe  model  for
                        datasets too small for quadratic models (or  for  very
                        noisy problems).
                * 2     quadratic  model,  least  squares  fitting. Best model
                        available (if your dataset is large enough).
        NQ  -   number of points used to calculate nodal functions.  NQ should
                be  significantly   larger   than  1.5  times  the  number  of
                coefficients in a nodal function to overcome effects of noise:
                * larger than 1.5*(1+NX) for linear model,
                * larger than 3/4*(NX+2)*(NX+1) for quadratic model.
                Values less than this threshold will be silently increased.
        NW  -   number of points used to calculate weights and to interpolate.
                Required: >=2^NX+1, values less than this  threshold  will  be
                silently increased.
                Recommended value: about 2*NQ or larger

    OUTPUT PARAMETERS:
        Z   -   IDW interpolant.

    NOTES:
      * best results are obtained with quadratic models, linear models are not
        recommended to use unless you are pretty sure that it is what you want
      * this subroutine is always succeeds (as long as correct parameters  are
        passed).
      * see  'Multivariate  Interpolation  of Large Sets of Scattered Data' by
        Robert J. Renka for more information on this algorithm.


      -- ALGLIB PROJECT --
         Copyright 02.03.2010 by Bochkanov Sergey
    *************************************************************************/
    public static void idwbuildnoisy(double[,] xy, int n, int nx, int d, int nq, int nw, out idwinterpolant z)
    {
        z = new idwinterpolant();
        idwint.idwbuildnoisy(xy, n, nx, d, nq, nw, z.innerobj);
        return;
    }

}
public partial class alglib
{


    /*************************************************************************
    Barycentric interpolant.
    *************************************************************************/
    public class barycentricinterpolant
    {
        //
        // Public declarations
        //

        public barycentricinterpolant()
        {
            _innerobj = new ratint.barycentricinterpolant();
        }

        //
        // Although some of declarations below are public, you should not use them
        // They are intended for internal use only
        //
        private ratint.barycentricinterpolant _innerobj;
        public ratint.barycentricinterpolant innerobj { get { return _innerobj; } }
        public barycentricinterpolant(ratint.barycentricinterpolant obj)
        {
            _innerobj = obj;
        }
    }

    /*************************************************************************
    Rational interpolation using barycentric formula

    F(t) = SUM(i=0,n-1,w[i]*f[i]/(t-x[i])) / SUM(i=0,n-1,w[i]/(t-x[i]))

    Input parameters:
        B   -   barycentric interpolant built with one of model building
                subroutines.
        T   -   interpolation point

    Result:
        barycentric interpolant F(t)

      -- ALGLIB --
         Copyright 17.08.2009 by Bochkanov Sergey
    *************************************************************************/
    public static double barycentriccalc(barycentricinterpolant b, double t)
    {

        double result = ratint.barycentriccalc(b.innerobj, t);
        return result;
    }

    /*************************************************************************
    Differentiation of barycentric interpolant: first derivative.

    Algorithm used in this subroutine is very robust and should not fail until
    provided with values too close to MaxRealNumber  (usually  MaxRealNumber/N
    or greater will overflow).

    INPUT PARAMETERS:
        B   -   barycentric interpolant built with one of model building
                subroutines.
        T   -   interpolation point

    OUTPUT PARAMETERS:
        F   -   barycentric interpolant at T
        DF  -   first derivative

    NOTE


      -- ALGLIB --
         Copyright 17.08.2009 by Bochkanov Sergey
    *************************************************************************/
    public static void barycentricdiff1(barycentricinterpolant b, double t, out double f, out double df)
    {
        f = 0;
        df = 0;
        ratint.barycentricdiff1(b.innerobj, t, ref f, ref df);
        return;
    }

    /*************************************************************************
    Differentiation of barycentric interpolant: first/second derivatives.

    INPUT PARAMETERS:
        B   -   barycentric interpolant built with one of model building
                subroutines.
        T   -   interpolation point

    OUTPUT PARAMETERS:
        F   -   barycentric interpolant at T
        DF  -   first derivative
        D2F -   second derivative

    NOTE: this algorithm may fail due to overflow/underflor if  used  on  data
    whose values are close to MaxRealNumber or MinRealNumber.  Use more robust
    BarycentricDiff1() subroutine in such cases.


      -- ALGLIB --
         Copyright 17.08.2009 by Bochkanov Sergey
    *************************************************************************/
    public static void barycentricdiff2(barycentricinterpolant b, double t, out double f, out double df, out double d2f)
    {
        f = 0;
        df = 0;
        d2f = 0;
        ratint.barycentricdiff2(b.innerobj, t, ref f, ref df, ref d2f);
        return;
    }

    /*************************************************************************
    This subroutine performs linear transformation of the argument.

    INPUT PARAMETERS:
        B       -   rational interpolant in barycentric form
        CA, CB  -   transformation coefficients: x = CA*t + CB

    OUTPUT PARAMETERS:
        B       -   transformed interpolant with X replaced by T

      -- ALGLIB PROJECT --
         Copyright 19.08.2009 by Bochkanov Sergey
    *************************************************************************/
    public static void barycentriclintransx(barycentricinterpolant b, double ca, double cb)
    {

        ratint.barycentriclintransx(b.innerobj, ca, cb);
        return;
    }

    /*************************************************************************
    This  subroutine   performs   linear  transformation  of  the  barycentric
    interpolant.

    INPUT PARAMETERS:
        B       -   rational interpolant in barycentric form
        CA, CB  -   transformation coefficients: B2(x) = CA*B(x) + CB

    OUTPUT PARAMETERS:
        B       -   transformed interpolant

      -- ALGLIB PROJECT --
         Copyright 19.08.2009 by Bochkanov Sergey
    *************************************************************************/
    public static void barycentriclintransy(barycentricinterpolant b, double ca, double cb)
    {

        ratint.barycentriclintransy(b.innerobj, ca, cb);
        return;
    }

    /*************************************************************************
    Extracts X/Y/W arrays from rational interpolant

    INPUT PARAMETERS:
        B   -   barycentric interpolant

    OUTPUT PARAMETERS:
        N   -   nodes count, N>0
        X   -   interpolation nodes, array[0..N-1]
        F   -   function values, array[0..N-1]
        W   -   barycentric weights, array[0..N-1]

      -- ALGLIB --
         Copyright 17.08.2009 by Bochkanov Sergey
    *************************************************************************/
    public static void barycentricunpack(barycentricinterpolant b, out int n, out double[] x, out double[] y, out double[] w)
    {
        n = 0;
        x = new double[0];
        y = new double[0];
        w = new double[0];
        ratint.barycentricunpack(b.innerobj, ref n, ref x, ref y, ref w);
        return;
    }

    /*************************************************************************
    Rational interpolant from X/Y/W arrays

    F(t) = SUM(i=0,n-1,w[i]*f[i]/(t-x[i])) / SUM(i=0,n-1,w[i]/(t-x[i]))

    INPUT PARAMETERS:
        X   -   interpolation nodes, array[0..N-1]
        F   -   function values, array[0..N-1]
        W   -   barycentric weights, array[0..N-1]
        N   -   nodes count, N>0

    OUTPUT PARAMETERS:
        B   -   barycentric interpolant built from (X, Y, W)

      -- ALGLIB --
         Copyright 17.08.2009 by Bochkanov Sergey
    *************************************************************************/
    public static void barycentricbuildxyw(double[] x, double[] y, double[] w, int n, out barycentricinterpolant b)
    {
        b = new barycentricinterpolant();
        ratint.barycentricbuildxyw(x, y, w, n, b.innerobj);
        return;
    }

    /*************************************************************************
    Rational interpolant without poles

    The subroutine constructs the rational interpolating function without real
    poles  (see  'Barycentric rational interpolation with no  poles  and  high
    rates of approximation', Michael S. Floater. and  Kai  Hormann,  for  more
    information on this subject).

    Input parameters:
        X   -   interpolation nodes, array[0..N-1].
        Y   -   function values, array[0..N-1].
        N   -   number of nodes, N>0.
        D   -   order of the interpolation scheme, 0 <= D <= N-1.
                D<0 will cause an error.
                D>=N it will be replaced with D=N-1.
                if you don't know what D to choose, use small value about 3-5.

    Output parameters:
        B   -   barycentric interpolant.

    Note:
        this algorithm always succeeds and calculates the weights  with  close
        to machine precision.

      -- ALGLIB PROJECT --
         Copyright 17.06.2007 by Bochkanov Sergey
    *************************************************************************/
    public static void barycentricbuildfloaterhormann(double[] x, double[] y, int n, int d, out barycentricinterpolant b)
    {
        b = new barycentricinterpolant();
        ratint.barycentricbuildfloaterhormann(x, y, n, d, b.innerobj);
        return;
    }

}
public partial class alglib
{


    /*************************************************************************
    Conversion from barycentric representation to Chebyshev basis.
    This function has O(N^2) complexity.

    INPUT PARAMETERS:
        P   -   polynomial in barycentric form
        A,B -   base interval for Chebyshev polynomials (see below)
                A<>B

    OUTPUT PARAMETERS
        T   -   coefficients of Chebyshev representation;
                P(x) = sum { T[i]*Ti(2*(x-A)/(B-A)-1), i=0..N-1 },
                where Ti - I-th Chebyshev polynomial.

    NOTES:
        barycentric interpolant passed as P may be either polynomial  obtained
        from  polynomial  interpolation/ fitting or rational function which is
        NOT polynomial. We can't distinguish between these two cases, and this
        algorithm just tries to work assuming that P IS a polynomial.  If not,
        algorithm will return results, but they won't have any meaning.

      -- ALGLIB --
         Copyright 30.09.2010 by Bochkanov Sergey
    *************************************************************************/
    public static void polynomialbar2cheb(barycentricinterpolant p, double a, double b, out double[] t)
    {
        t = new double[0];
        polint.polynomialbar2cheb(p.innerobj, a, b, ref t);
        return;
    }

    /*************************************************************************
    Conversion from Chebyshev basis to barycentric representation.
    This function has O(N^2) complexity.

    INPUT PARAMETERS:
        T   -   coefficients of Chebyshev representation;
                P(x) = sum { T[i]*Ti(2*(x-A)/(B-A)-1), i=0..N },
                where Ti - I-th Chebyshev polynomial.
        N   -   number of coefficients:
                * if given, only leading N elements of T are used
                * if not given, automatically determined from size of T
        A,B -   base interval for Chebyshev polynomials (see above)
                A<B

    OUTPUT PARAMETERS
        P   -   polynomial in barycentric form

      -- ALGLIB --
         Copyright 30.09.2010 by Bochkanov Sergey
    *************************************************************************/
    public static void polynomialcheb2bar(double[] t, int n, double a, double b, out barycentricinterpolant p)
    {
        p = new barycentricinterpolant();
        polint.polynomialcheb2bar(t, n, a, b, p.innerobj);
        return;
    }
    public static void polynomialcheb2bar(double[] t, double a, double b, out barycentricinterpolant p)
    {
        int n;

        p = new barycentricinterpolant();
        n = ap.len(t);
        polint.polynomialcheb2bar(t, n, a, b, p.innerobj);

        return;
    }

    /*************************************************************************
    Conversion from barycentric representation to power basis.
    This function has O(N^2) complexity.

    INPUT PARAMETERS:
        P   -   polynomial in barycentric form
        C   -   offset (see below); 0.0 is used as default value.
        S   -   scale (see below);  1.0 is used as default value. S<>0.

    OUTPUT PARAMETERS
        A   -   coefficients, P(x) = sum { A[i]*((X-C)/S)^i, i=0..N-1 }
        N   -   number of coefficients (polynomial degree plus 1)

    NOTES:
    1.  this function accepts offset and scale, which can be  set  to  improve
        numerical properties of polynomial. For example, if P was obtained  as
        result of interpolation on [-1,+1],  you  can  set  C=0  and  S=1  and
        represent  P  as sum of 1, x, x^2, x^3 and so on. In most cases you it
        is exactly what you need.

        However, if your interpolation model was built on [999,1001], you will
        see significant growth of numerical errors when using {1, x, x^2, x^3}
        as basis. Representing P as sum of 1, (x-1000), (x-1000)^2, (x-1000)^3
        will be better option. Such representation can be  obtained  by  using
        1000.0 as offset C and 1.0 as scale S.

    2.  power basis is ill-conditioned and tricks described above can't  solve
        this problem completely. This function  will  return  coefficients  in
        any  case,  but  for  N>8  they  will  become unreliable. However, N's
        less than 5 are pretty safe.

    3.  barycentric interpolant passed as P may be either polynomial  obtained
        from  polynomial  interpolation/ fitting or rational function which is
        NOT polynomial. We can't distinguish between these two cases, and this
        algorithm just tries to work assuming that P IS a polynomial.  If not,
        algorithm will return results, but they won't have any meaning.

      -- ALGLIB --
         Copyright 30.09.2010 by Bochkanov Sergey
    *************************************************************************/
    public static void polynomialbar2pow(barycentricinterpolant p, double c, double s, out double[] a)
    {
        a = new double[0];
        polint.polynomialbar2pow(p.innerobj, c, s, ref a);
        return;
    }
    public static void polynomialbar2pow(barycentricinterpolant p, out double[] a)
    {
        double c;
        double s;

        a = new double[0];
        c = 0;
        s = 1;
        polint.polynomialbar2pow(p.innerobj, c, s, ref a);

        return;
    }

    /*************************************************************************
    Conversion from power basis to barycentric representation.
    This function has O(N^2) complexity.

    INPUT PARAMETERS:
        A   -   coefficients, P(x) = sum { A[i]*((X-C)/S)^i, i=0..N-1 }
        N   -   number of coefficients (polynomial degree plus 1)
                * if given, only leading N elements of A are used
                * if not given, automatically determined from size of A
        C   -   offset (see below); 0.0 is used as default value.
        S   -   scale (see below);  1.0 is used as default value. S<>0.

    OUTPUT PARAMETERS
        P   -   polynomial in barycentric form


    NOTES:
    1.  this function accepts offset and scale, which can be  set  to  improve
        numerical properties of polynomial. For example, if you interpolate on
        [-1,+1],  you  can  set C=0 and S=1 and convert from sum of 1, x, x^2,
        x^3 and so on. In most cases you it is exactly what you need.

        However, if your interpolation model was built on [999,1001], you will
        see significant growth of numerical errors when using {1, x, x^2, x^3}
        as  input  basis.  Converting  from  sum  of  1, (x-1000), (x-1000)^2,
        (x-1000)^3 will be better option (you have to specify 1000.0 as offset
        C and 1.0 as scale S).

    2.  power basis is ill-conditioned and tricks described above can't  solve
        this problem completely. This function  will  return barycentric model
        in any case, but for N>8 accuracy well degrade. However, N's less than
        5 are pretty safe.

      -- ALGLIB --
         Copyright 30.09.2010 by Bochkanov Sergey
    *************************************************************************/
    public static void polynomialpow2bar(double[] a, int n, double c, double s, out barycentricinterpolant p)
    {
        p = new barycentricinterpolant();
        polint.polynomialpow2bar(a, n, c, s, p.innerobj);
        return;
    }
    public static void polynomialpow2bar(double[] a, out barycentricinterpolant p)
    {
        int n;
        double c;
        double s;

        p = new barycentricinterpolant();
        n = ap.len(a);
        c = 0;
        s = 1;
        polint.polynomialpow2bar(a, n, c, s, p.innerobj);

        return;
    }

    /*************************************************************************
    Lagrange intepolant: generation of the model on the general grid.
    This function has O(N^2) complexity.

    INPUT PARAMETERS:
        X   -   abscissas, array[0..N-1]
        Y   -   function values, array[0..N-1]
        N   -   number of points, N>=1

    OUTPUT PARAMETERS
        P   -   barycentric model which represents Lagrange interpolant
                (see ratint unit info and BarycentricCalc() description for
                more information).

      -- ALGLIB --
         Copyright 02.12.2009 by Bochkanov Sergey
    *************************************************************************/
    public static void polynomialbuild(double[] x, double[] y, int n, out barycentricinterpolant p)
    {
        p = new barycentricinterpolant();
        polint.polynomialbuild(x, y, n, p.innerobj);
        return;
    }
    public static void polynomialbuild(double[] x, double[] y, out barycentricinterpolant p)
    {
        int n;
        if( (ap.len(x)!=ap.len(y)))
            throw new alglibexception("Error while calling 'polynomialbuild': looks like one of arguments has wrong size");
        p = new barycentricinterpolant();
        n = ap.len(x);
        polint.polynomialbuild(x, y, n, p.innerobj);

        return;
    }

    /*************************************************************************
    Lagrange intepolant: generation of the model on equidistant grid.
    This function has O(N) complexity.

    INPUT PARAMETERS:
        A   -   left boundary of [A,B]
        B   -   right boundary of [A,B]
        Y   -   function values at the nodes, array[0..N-1]
        N   -   number of points, N>=1
                for N=1 a constant model is constructed.

    OUTPUT PARAMETERS
        P   -   barycentric model which represents Lagrange interpolant
                (see ratint unit info and BarycentricCalc() description for
                more information).

      -- ALGLIB --
         Copyright 03.12.2009 by Bochkanov Sergey
    *************************************************************************/
    public static void polynomialbuildeqdist(double a, double b, double[] y, int n, out barycentricinterpolant p)
    {
        p = new barycentricinterpolant();
        polint.polynomialbuildeqdist(a, b, y, n, p.innerobj);
        return;
    }
    public static void polynomialbuildeqdist(double a, double b, double[] y, out barycentricinterpolant p)
    {
        int n;

        p = new barycentricinterpolant();
        n = ap.len(y);
        polint.polynomialbuildeqdist(a, b, y, n, p.innerobj);

        return;
    }

    /*************************************************************************
    Lagrange intepolant on Chebyshev grid (first kind).
    This function has O(N) complexity.

    INPUT PARAMETERS:
        A   -   left boundary of [A,B]
        B   -   right boundary of [A,B]
        Y   -   function values at the nodes, array[0..N-1],
                Y[I] = Y(0.5*(B+A) + 0.5*(B-A)*Cos(PI*(2*i+1)/(2*n)))
        N   -   number of points, N>=1
                for N=1 a constant model is constructed.

    OUTPUT PARAMETERS
        P   -   barycentric model which represents Lagrange interpolant
                (see ratint unit info and BarycentricCalc() description for
                more information).

      -- ALGLIB --
         Copyright 03.12.2009 by Bochkanov Sergey
    *************************************************************************/
    public static void polynomialbuildcheb1(double a, double b, double[] y, int n, out barycentricinterpolant p)
    {
        p = new barycentricinterpolant();
        polint.polynomialbuildcheb1(a, b, y, n, p.innerobj);
        return;
    }
    public static void polynomialbuildcheb1(double a, double b, double[] y, out barycentricinterpolant p)
    {
        int n;

        p = new barycentricinterpolant();
        n = ap.len(y);
        polint.polynomialbuildcheb1(a, b, y, n, p.innerobj);

        return;
    }

    /*************************************************************************
    Lagrange intepolant on Chebyshev grid (second kind).
    This function has O(N) complexity.

    INPUT PARAMETERS:
        A   -   left boundary of [A,B]
        B   -   right boundary of [A,B]
        Y   -   function values at the nodes, array[0..N-1],
                Y[I] = Y(0.5*(B+A) + 0.5*(B-A)*Cos(PI*i/(n-1)))
        N   -   number of points, N>=1
                for N=1 a constant model is constructed.

    OUTPUT PARAMETERS
        P   -   barycentric model which represents Lagrange interpolant
                (see ratint unit info and BarycentricCalc() description for
                more information).

      -- ALGLIB --
         Copyright 03.12.2009 by Bochkanov Sergey
    *************************************************************************/
    public static void polynomialbuildcheb2(double a, double b, double[] y, int n, out barycentricinterpolant p)
    {
        p = new barycentricinterpolant();
        polint.polynomialbuildcheb2(a, b, y, n, p.innerobj);
        return;
    }
    public static void polynomialbuildcheb2(double a, double b, double[] y, out barycentricinterpolant p)
    {
        int n;

        p = new barycentricinterpolant();
        n = ap.len(y);
        polint.polynomialbuildcheb2(a, b, y, n, p.innerobj);

        return;
    }

    /*************************************************************************
    Fast equidistant polynomial interpolation function with O(N) complexity

    INPUT PARAMETERS:
        A   -   left boundary of [A,B]
        B   -   right boundary of [A,B]
        F   -   function values, array[0..N-1]
        N   -   number of points on equidistant grid, N>=1
                for N=1 a constant model is constructed.
        T   -   position where P(x) is calculated

    RESULT
        value of the Lagrange interpolant at T

    IMPORTANT
        this function provides fast interface which is not overflow-safe
        nor it is very precise.
        the best option is to use  PolynomialBuildEqDist()/BarycentricCalc()
        subroutines unless you are pretty sure that your data will not result
        in overflow.

      -- ALGLIB --
         Copyright 02.12.2009 by Bochkanov Sergey
    *************************************************************************/
    public static double polynomialcalceqdist(double a, double b, double[] f, int n, double t)
    {

        double result = polint.polynomialcalceqdist(a, b, f, n, t);
        return result;
    }
    public static double polynomialcalceqdist(double a, double b, double[] f, double t)
    {
        int n;


        n = ap.len(f);
        double result = polint.polynomialcalceqdist(a, b, f, n, t);

        return result;
    }

    /*************************************************************************
    Fast polynomial interpolation function on Chebyshev points (first kind)
    with O(N) complexity.

    INPUT PARAMETERS:
        A   -   left boundary of [A,B]
        B   -   right boundary of [A,B]
        F   -   function values, array[0..N-1]
        N   -   number of points on Chebyshev grid (first kind),
                X[i] = 0.5*(B+A) + 0.5*(B-A)*Cos(PI*(2*i+1)/(2*n))
                for N=1 a constant model is constructed.
        T   -   position where P(x) is calculated

    RESULT
        value of the Lagrange interpolant at T

    IMPORTANT
        this function provides fast interface which is not overflow-safe
        nor it is very precise.
        the best option is to use  PolIntBuildCheb1()/BarycentricCalc()
        subroutines unless you are pretty sure that your data will not result
        in overflow.

      -- ALGLIB --
         Copyright 02.12.2009 by Bochkanov Sergey
    *************************************************************************/
    public static double polynomialcalccheb1(double a, double b, double[] f, int n, double t)
    {

        double result = polint.polynomialcalccheb1(a, b, f, n, t);
        return result;
    }
    public static double polynomialcalccheb1(double a, double b, double[] f, double t)
    {
        int n;


        n = ap.len(f);
        double result = polint.polynomialcalccheb1(a, b, f, n, t);

        return result;
    }

    /*************************************************************************
    Fast polynomial interpolation function on Chebyshev points (second kind)
    with O(N) complexity.

    INPUT PARAMETERS:
        A   -   left boundary of [A,B]
        B   -   right boundary of [A,B]
        F   -   function values, array[0..N-1]
        N   -   number of points on Chebyshev grid (second kind),
                X[i] = 0.5*(B+A) + 0.5*(B-A)*Cos(PI*i/(n-1))
                for N=1 a constant model is constructed.
        T   -   position where P(x) is calculated

    RESULT
        value of the Lagrange interpolant at T

    IMPORTANT
        this function provides fast interface which is not overflow-safe
        nor it is very precise.
        the best option is to use PolIntBuildCheb2()/BarycentricCalc()
        subroutines unless you are pretty sure that your data will not result
        in overflow.

      -- ALGLIB --
         Copyright 02.12.2009 by Bochkanov Sergey
    *************************************************************************/
    public static double polynomialcalccheb2(double a, double b, double[] f, int n, double t)
    {

        double result = polint.polynomialcalccheb2(a, b, f, n, t);
        return result;
    }
    public static double polynomialcalccheb2(double a, double b, double[] f, double t)
    {
        int n;


        n = ap.len(f);
        double result = polint.polynomialcalccheb2(a, b, f, n, t);

        return result;
    }

}
public partial class alglib
{


    /*************************************************************************
    1-dimensional spline interpolant
    *************************************************************************/
    public class spline1dinterpolant
    {
        //
        // Public declarations
        //

        public spline1dinterpolant()
        {
            _innerobj = new spline1d.spline1dinterpolant();
        }

        //
        // Although some of declarations below are public, you should not use them
        // They are intended for internal use only
        //
        private spline1d.spline1dinterpolant _innerobj;
        public spline1d.spline1dinterpolant innerobj { get { return _innerobj; } }
        public spline1dinterpolant(spline1d.spline1dinterpolant obj)
        {
            _innerobj = obj;
        }
    }

    /*************************************************************************
    This subroutine builds linear spline interpolant

    INPUT PARAMETERS:
        X   -   spline nodes, array[0..N-1]
        Y   -   function values, array[0..N-1]
        N   -   points count (optional):
                * N>=2
                * if given, only first N points are used to build spline
                * if not given, automatically detected from X/Y sizes
                  (len(X) must be equal to len(Y))

    OUTPUT PARAMETERS:
        C   -   spline interpolant


    ORDER OF POINTS

    Subroutine automatically sorts points, so caller may pass unsorted array.

      -- ALGLIB PROJECT --
         Copyright 24.06.2007 by Bochkanov Sergey
    *************************************************************************/
    public static void spline1dbuildlinear(double[] x, double[] y, int n, out spline1dinterpolant c)
    {
        c = new spline1dinterpolant();
        spline1d.spline1dbuildlinear(x, y, n, c.innerobj);
        return;
    }
    public static void spline1dbuildlinear(double[] x, double[] y, out spline1dinterpolant c)
    {
        int n;
        if( (ap.len(x)!=ap.len(y)))
            throw new alglibexception("Error while calling 'spline1dbuildlinear': looks like one of arguments has wrong size");
        c = new spline1dinterpolant();
        n = ap.len(x);
        spline1d.spline1dbuildlinear(x, y, n, c.innerobj);

        return;
    }

    /*************************************************************************
    This subroutine builds cubic spline interpolant.

    INPUT PARAMETERS:
        X           -   spline nodes, array[0..N-1].
        Y           -   function values, array[0..N-1].

    OPTIONAL PARAMETERS:
        N           -   points count:
                        * N>=2
                        * if given, only first N points are used to build spline
                        * if not given, automatically detected from X/Y sizes
                          (len(X) must be equal to len(Y))
        BoundLType  -   boundary condition type for the left boundary
        BoundL      -   left boundary condition (first or second derivative,
                        depending on the BoundLType)
        BoundRType  -   boundary condition type for the right boundary
        BoundR      -   right boundary condition (first or second derivative,
                        depending on the BoundRType)

    OUTPUT PARAMETERS:
        C           -   spline interpolant

    ORDER OF POINTS

    Subroutine automatically sorts points, so caller may pass unsorted array.

    SETTING BOUNDARY VALUES:

    The BoundLType/BoundRType parameters can have the following values:
        * -1, which corresonds to the periodic (cyclic) boundary conditions.
              In this case:
              * both BoundLType and BoundRType must be equal to -1.
              * BoundL/BoundR are ignored
              * Y[last] is ignored (it is assumed to be equal to Y[first]).
        *  0, which  corresponds  to  the  parabolically   terminated  spline
              (BoundL and/or BoundR are ignored).
        *  1, which corresponds to the first derivative boundary condition
        *  2, which corresponds to the second derivative boundary condition
        *  by default, BoundType=0 is used

    PROBLEMS WITH PERIODIC BOUNDARY CONDITIONS:

    Problems with periodic boundary conditions have Y[first_point]=Y[last_point].
    However, this subroutine doesn't require you to specify equal  values  for
    the first and last points - it automatically forces them  to  be  equal by
    copying  Y[first_point]  (corresponds  to the leftmost,  minimal  X[])  to
    Y[last_point]. However it is recommended to pass consistent values of Y[],
    i.e. to make Y[first_point]=Y[last_point].

      -- ALGLIB PROJECT --
         Copyright 23.06.2007 by Bochkanov Sergey
    *************************************************************************/
    public static void spline1dbuildcubic(double[] x, double[] y, int n, int boundltype, double boundl, int boundrtype, double boundr, out spline1dinterpolant c)
    {
        c = new spline1dinterpolant();
        spline1d.spline1dbuildcubic(x, y, n, boundltype, boundl, boundrtype, boundr, c.innerobj);
        return;
    }
    public static void spline1dbuildcubic(double[] x, double[] y, out spline1dinterpolant c)
    {
        int n;
        int boundltype;
        double boundl;
        int boundrtype;
        double boundr;
        if( (ap.len(x)!=ap.len(y)))
            throw new alglibexception("Error while calling 'spline1dbuildcubic': looks like one of arguments has wrong size");
        c = new spline1dinterpolant();
        n = ap.len(x);
        boundltype = 0;
        boundl = 0;
        boundrtype = 0;
        boundr = 0;
        spline1d.spline1dbuildcubic(x, y, n, boundltype, boundl, boundrtype, boundr, c.innerobj);

        return;
    }

    /*************************************************************************
    This function solves following problem: given table y[] of function values
    at nodes x[], it calculates and returns table of function derivatives  d[]
    (calculated at the same nodes x[]).

    This function yields same result as Spline1DBuildCubic() call followed  by
    sequence of Spline1DDiff() calls, but it can be several times faster  when
    called for ordered X[] and X2[].

    INPUT PARAMETERS:
        X           -   spline nodes
        Y           -   function values

    OPTIONAL PARAMETERS:
        N           -   points count:
                        * N>=2
                        * if given, only first N points are used
                        * if not given, automatically detected from X/Y sizes
                          (len(X) must be equal to len(Y))
        BoundLType  -   boundary condition type for the left boundary
        BoundL      -   left boundary condition (first or second derivative,
                        depending on the BoundLType)
        BoundRType  -   boundary condition type for the right boundary
        BoundR      -   right boundary condition (first or second derivative,
                        depending on the BoundRType)

    OUTPUT PARAMETERS:
        D           -   derivative values at X[]

    ORDER OF POINTS

    Subroutine automatically sorts points, so caller may pass unsorted array.
    Derivative values are correctly reordered on return, so  D[I]  is  always
    equal to S'(X[I]) independently of points order.

    SETTING BOUNDARY VALUES:

    The BoundLType/BoundRType parameters can have the following values:
        * -1, which corresonds to the periodic (cyclic) boundary conditions.
              In this case:
              * both BoundLType and BoundRType must be equal to -1.
              * BoundL/BoundR are ignored
              * Y[last] is ignored (it is assumed to be equal to Y[first]).
        *  0, which  corresponds  to  the  parabolically   terminated  spline
              (BoundL and/or BoundR are ignored).
        *  1, which corresponds to the first derivative boundary condition
        *  2, which corresponds to the second derivative boundary condition
        *  by default, BoundType=0 is used

    PROBLEMS WITH PERIODIC BOUNDARY CONDITIONS:

    Problems with periodic boundary conditions have Y[first_point]=Y[last_point].
    However, this subroutine doesn't require you to specify equal  values  for
    the first and last points - it automatically forces them  to  be  equal by
    copying  Y[first_point]  (corresponds  to the leftmost,  minimal  X[])  to
    Y[last_point]. However it is recommended to pass consistent values of Y[],
    i.e. to make Y[first_point]=Y[last_point].

      -- ALGLIB PROJECT --
         Copyright 03.09.2010 by Bochkanov Sergey
    *************************************************************************/
    public static void spline1dgriddiffcubic(double[] x, double[] y, int n, int boundltype, double boundl, int boundrtype, double boundr, out double[] d)
    {
        d = new double[0];
        spline1d.spline1dgriddiffcubic(x, y, n, boundltype, boundl, boundrtype, boundr, ref d);
        return;
    }
    public static void spline1dgriddiffcubic(double[] x, double[] y, out double[] d)
    {
        int n;
        int boundltype;
        double boundl;
        int boundrtype;
        double boundr;
        if( (ap.len(x)!=ap.len(y)))
            throw new alglibexception("Error while calling 'spline1dgriddiffcubic': looks like one of arguments has wrong size");
        d = new double[0];
        n = ap.len(x);
        boundltype = 0;
        boundl = 0;
        boundrtype = 0;
        boundr = 0;
        spline1d.spline1dgriddiffcubic(x, y, n, boundltype, boundl, boundrtype, boundr, ref d);

        return;
    }

    /*************************************************************************
    This function solves following problem: given table y[] of function values
    at  nodes  x[],  it  calculates  and  returns  tables  of first and second
    function derivatives d1[] and d2[] (calculated at the same nodes x[]).

    This function yields same result as Spline1DBuildCubic() call followed  by
    sequence of Spline1DDiff() calls, but it can be several times faster  when
    called for ordered X[] and X2[].

    INPUT PARAMETERS:
        X           -   spline nodes
        Y           -   function values

    OPTIONAL PARAMETERS:
        N           -   points count:
                        * N>=2
                        * if given, only first N points are used
                        * if not given, automatically detected from X/Y sizes
                          (len(X) must be equal to len(Y))
        BoundLType  -   boundary condition type for the left boundary
        BoundL      -   left boundary condition (first or second derivative,
                        depending on the BoundLType)
        BoundRType  -   boundary condition type for the right boundary
        BoundR      -   right boundary condition (first or second derivative,
                        depending on the BoundRType)

    OUTPUT PARAMETERS:
        D1          -   S' values at X[]
        D2          -   S'' values at X[]

    ORDER OF POINTS

    Subroutine automatically sorts points, so caller may pass unsorted array.
    Derivative values are correctly reordered on return, so  D[I]  is  always
    equal to S'(X[I]) independently of points order.

    SETTING BOUNDARY VALUES:

    The BoundLType/BoundRType parameters can have the following values:
        * -1, which corresonds to the periodic (cyclic) boundary conditions.
              In this case:
              * both BoundLType and BoundRType must be equal to -1.
              * BoundL/BoundR are ignored
              * Y[last] is ignored (it is assumed to be equal to Y[first]).
        *  0, which  corresponds  to  the  parabolically   terminated  spline
              (BoundL and/or BoundR are ignored).
        *  1, which corresponds to the first derivative boundary condition
        *  2, which corresponds to the second derivative boundary condition
        *  by default, BoundType=0 is used

    PROBLEMS WITH PERIODIC BOUNDARY CONDITIONS:

    Problems with periodic boundary conditions have Y[first_point]=Y[last_point].
    However, this subroutine doesn't require you to specify equal  values  for
    the first and last points - it automatically forces them  to  be  equal by
    copying  Y[first_point]  (corresponds  to the leftmost,  minimal  X[])  to
    Y[last_point]. However it is recommended to pass consistent values of Y[],
    i.e. to make Y[first_point]=Y[last_point].

      -- ALGLIB PROJECT --
         Copyright 03.09.2010 by Bochkanov Sergey
    *************************************************************************/
    public static void spline1dgriddiff2cubic(double[] x, double[] y, int n, int boundltype, double boundl, int boundrtype, double boundr, out double[] d1, out double[] d2)
    {
        d1 = new double[0];
        d2 = new double[0];
        spline1d.spline1dgriddiff2cubic(x, y, n, boundltype, boundl, boundrtype, boundr, ref d1, ref d2);
        return;
    }
    public static void spline1dgriddiff2cubic(double[] x, double[] y, out double[] d1, out double[] d2)
    {
        int n;
        int boundltype;
        double boundl;
        int boundrtype;
        double boundr;
        if( (ap.len(x)!=ap.len(y)))
            throw new alglibexception("Error while calling 'spline1dgriddiff2cubic': looks like one of arguments has wrong size");
        d1 = new double[0];
        d2 = new double[0];
        n = ap.len(x);
        boundltype = 0;
        boundl = 0;
        boundrtype = 0;
        boundr = 0;
        spline1d.spline1dgriddiff2cubic(x, y, n, boundltype, boundl, boundrtype, boundr, ref d1, ref d2);

        return;
    }

    /*************************************************************************
    This function solves following problem: given table y[] of function values
    at old nodes x[]  and new nodes  x2[],  it calculates and returns table of
    function values y2[] (calculated at x2[]).

    This function yields same result as Spline1DBuildCubic() call followed  by
    sequence of Spline1DDiff() calls, but it can be several times faster  when
    called for ordered X[] and X2[].

    INPUT PARAMETERS:
        X           -   old spline nodes
        Y           -   function values
        X2           -  new spline nodes

    OPTIONAL PARAMETERS:
        N           -   points count:
                        * N>=2
                        * if given, only first N points from X/Y are used
                        * if not given, automatically detected from X/Y sizes
                          (len(X) must be equal to len(Y))
        BoundLType  -   boundary condition type for the left boundary
        BoundL      -   left boundary condition (first or second derivative,
                        depending on the BoundLType)
        BoundRType  -   boundary condition type for the right boundary
        BoundR      -   right boundary condition (first or second derivative,
                        depending on the BoundRType)
        N2          -   new points count:
                        * N2>=2
                        * if given, only first N2 points from X2 are used
                        * if not given, automatically detected from X2 size

    OUTPUT PARAMETERS:
        F2          -   function values at X2[]

    ORDER OF POINTS

    Subroutine automatically sorts points, so caller  may pass unsorted array.
    Function  values  are correctly reordered on  return, so F2[I]  is  always
    equal to S(X2[I]) independently of points order.

    SETTING BOUNDARY VALUES:

    The BoundLType/BoundRType parameters can have the following values:
        * -1, which corresonds to the periodic (cyclic) boundary conditions.
              In this case:
              * both BoundLType and BoundRType must be equal to -1.
              * BoundL/BoundR are ignored
              * Y[last] is ignored (it is assumed to be equal to Y[first]).
        *  0, which  corresponds  to  the  parabolically   terminated  spline
              (BoundL and/or BoundR are ignored).
        *  1, which corresponds to the first derivative boundary condition
        *  2, which corresponds to the second derivative boundary condition
        *  by default, BoundType=0 is used

    PROBLEMS WITH PERIODIC BOUNDARY CONDITIONS:

    Problems with periodic boundary conditions have Y[first_point]=Y[last_point].
    However, this subroutine doesn't require you to specify equal  values  for
    the first and last points - it automatically forces them  to  be  equal by
    copying  Y[first_point]  (corresponds  to the leftmost,  minimal  X[])  to
    Y[last_point]. However it is recommended to pass consistent values of Y[],
    i.e. to make Y[first_point]=Y[last_point].

      -- ALGLIB PROJECT --
         Copyright 03.09.2010 by Bochkanov Sergey
    *************************************************************************/
    public static void spline1dconvcubic(double[] x, double[] y, int n, int boundltype, double boundl, int boundrtype, double boundr, double[] x2, int n2, out double[] y2)
    {
        y2 = new double[0];
        spline1d.spline1dconvcubic(x, y, n, boundltype, boundl, boundrtype, boundr, x2, n2, ref y2);
        return;
    }
    public static void spline1dconvcubic(double[] x, double[] y, double[] x2, out double[] y2)
    {
        int n;
        int boundltype;
        double boundl;
        int boundrtype;
        double boundr;
        int n2;
        if( (ap.len(x)!=ap.len(y)))
            throw new alglibexception("Error while calling 'spline1dconvcubic': looks like one of arguments has wrong size");
        y2 = new double[0];
        n = ap.len(x);
        boundltype = 0;
        boundl = 0;
        boundrtype = 0;
        boundr = 0;
        n2 = ap.len(x2);
        spline1d.spline1dconvcubic(x, y, n, boundltype, boundl, boundrtype, boundr, x2, n2, ref y2);

        return;
    }

    /*************************************************************************
    This function solves following problem: given table y[] of function values
    at old nodes x[]  and new nodes  x2[],  it calculates and returns table of
    function values y2[] and derivatives d2[] (calculated at x2[]).

    This function yields same result as Spline1DBuildCubic() call followed  by
    sequence of Spline1DDiff() calls, but it can be several times faster  when
    called for ordered X[] and X2[].

    INPUT PARAMETERS:
        X           -   old spline nodes
        Y           -   function values
        X2           -  new spline nodes

    OPTIONAL PARAMETERS:
        N           -   points count:
                        * N>=2
                        * if given, only first N points from X/Y are used
                        * if not given, automatically detected from X/Y sizes
                          (len(X) must be equal to len(Y))
        BoundLType  -   boundary condition type for the left boundary
        BoundL      -   left boundary condition (first or second derivative,
                        depending on the BoundLType)
        BoundRType  -   boundary condition type for the right boundary
        BoundR      -   right boundary condition (first or second derivative,
                        depending on the BoundRType)
        N2          -   new points count:
                        * N2>=2
                        * if given, only first N2 points from X2 are used
                        * if not given, automatically detected from X2 size

    OUTPUT PARAMETERS:
        F2          -   function values at X2[]
        D2          -   first derivatives at X2[]

    ORDER OF POINTS

    Subroutine automatically sorts points, so caller  may pass unsorted array.
    Function  values  are correctly reordered on  return, so F2[I]  is  always
    equal to S(X2[I]) independently of points order.

    SETTING BOUNDARY VALUES:

    The BoundLType/BoundRType parameters can have the following values:
        * -1, which corresonds to the periodic (cyclic) boundary conditions.
              In this case:
              * both BoundLType and BoundRType must be equal to -1.
              * BoundL/BoundR are ignored
              * Y[last] is ignored (it is assumed to be equal to Y[first]).
        *  0, which  corresponds  to  the  parabolically   terminated  spline
              (BoundL and/or BoundR are ignored).
        *  1, which corresponds to the first derivative boundary condition
        *  2, which corresponds to the second derivative boundary condition
        *  by default, BoundType=0 is used

    PROBLEMS WITH PERIODIC BOUNDARY CONDITIONS:

    Problems with periodic boundary conditions have Y[first_point]=Y[last_point].
    However, this subroutine doesn't require you to specify equal  values  for
    the first and last points - it automatically forces them  to  be  equal by
    copying  Y[first_point]  (corresponds  to the leftmost,  minimal  X[])  to
    Y[last_point]. However it is recommended to pass consistent values of Y[],
    i.e. to make Y[first_point]=Y[last_point].

      -- ALGLIB PROJECT --
         Copyright 03.09.2010 by Bochkanov Sergey
    *************************************************************************/
    public static void spline1dconvdiffcubic(double[] x, double[] y, int n, int boundltype, double boundl, int boundrtype, double boundr, double[] x2, int n2, out double[] y2, out double[] d2)
    {
        y2 = new double[0];
        d2 = new double[0];
        spline1d.spline1dconvdiffcubic(x, y, n, boundltype, boundl, boundrtype, boundr, x2, n2, ref y2, ref d2);
        return;
    }
    public static void spline1dconvdiffcubic(double[] x, double[] y, double[] x2, out double[] y2, out double[] d2)
    {
        int n;
        int boundltype;
        double boundl;
        int boundrtype;
        double boundr;
        int n2;
        if( (ap.len(x)!=ap.len(y)))
            throw new alglibexception("Error while calling 'spline1dconvdiffcubic': looks like one of arguments has wrong size");
        y2 = new double[0];
        d2 = new double[0];
        n = ap.len(x);
        boundltype = 0;
        boundl = 0;
        boundrtype = 0;
        boundr = 0;
        n2 = ap.len(x2);
        spline1d.spline1dconvdiffcubic(x, y, n, boundltype, boundl, boundrtype, boundr, x2, n2, ref y2, ref d2);

        return;
    }

    /*************************************************************************
    This function solves following problem: given table y[] of function values
    at old nodes x[]  and new nodes  x2[],  it calculates and returns table of
    function  values  y2[],  first  and  second  derivatives  d2[]  and  dd2[]
    (calculated at x2[]).

    This function yields same result as Spline1DBuildCubic() call followed  by
    sequence of Spline1DDiff() calls, but it can be several times faster  when
    called for ordered X[] and X2[].

    INPUT PARAMETERS:
        X           -   old spline nodes
        Y           -   function values
        X2           -  new spline nodes

    OPTIONAL PARAMETERS:
        N           -   points count:
                        * N>=2
                        * if given, only first N points from X/Y are used
                        * if not given, automatically detected from X/Y sizes
                          (len(X) must be equal to len(Y))
        BoundLType  -   boundary condition type for the left boundary
        BoundL      -   left boundary condition (first or second derivative,
                        depending on the BoundLType)
        BoundRType  -   boundary condition type for the right boundary
        BoundR      -   right boundary condition (first or second derivative,
                        depending on the BoundRType)
        N2          -   new points count:
                        * N2>=2
                        * if given, only first N2 points from X2 are used
                        * if not given, automatically detected from X2 size

    OUTPUT PARAMETERS:
        F2          -   function values at X2[]
        D2          -   first derivatives at X2[]
        DD2         -   second derivatives at X2[]

    ORDER OF POINTS

    Subroutine automatically sorts points, so caller  may pass unsorted array.
    Function  values  are correctly reordered on  return, so F2[I]  is  always
    equal to S(X2[I]) independently of points order.

    SETTING BOUNDARY VALUES:

    The BoundLType/BoundRType parameters can have the following values:
        * -1, which corresonds to the periodic (cyclic) boundary conditions.
              In this case:
              * both BoundLType and BoundRType must be equal to -1.
              * BoundL/BoundR are ignored
              * Y[last] is ignored (it is assumed to be equal to Y[first]).
        *  0, which  corresponds  to  the  parabolically   terminated  spline
              (BoundL and/or BoundR are ignored).
        *  1, which corresponds to the first derivative boundary condition
        *  2, which corresponds to the second derivative boundary condition
        *  by default, BoundType=0 is used

    PROBLEMS WITH PERIODIC BOUNDARY CONDITIONS:

    Problems with periodic boundary conditions have Y[first_point]=Y[last_point].
    However, this subroutine doesn't require you to specify equal  values  for
    the first and last points - it automatically forces them  to  be  equal by
    copying  Y[first_point]  (corresponds  to the leftmost,  minimal  X[])  to
    Y[last_point]. However it is recommended to pass consistent values of Y[],
    i.e. to make Y[first_point]=Y[last_point].

      -- ALGLIB PROJECT --
         Copyright 03.09.2010 by Bochkanov Sergey
    *************************************************************************/
    public static void spline1dconvdiff2cubic(double[] x, double[] y, int n, int boundltype, double boundl, int boundrtype, double boundr, double[] x2, int n2, out double[] y2, out double[] d2, out double[] dd2)
    {
        y2 = new double[0];
        d2 = new double[0];
        dd2 = new double[0];
        spline1d.spline1dconvdiff2cubic(x, y, n, boundltype, boundl, boundrtype, boundr, x2, n2, ref y2, ref d2, ref dd2);
        return;
    }
    public static void spline1dconvdiff2cubic(double[] x, double[] y, double[] x2, out double[] y2, out double[] d2, out double[] dd2)
    {
        int n;
        int boundltype;
        double boundl;
        int boundrtype;
        double boundr;
        int n2;
        if( (ap.len(x)!=ap.len(y)))
            throw new alglibexception("Error while calling 'spline1dconvdiff2cubic': looks like one of arguments has wrong size");
        y2 = new double[0];
        d2 = new double[0];
        dd2 = new double[0];
        n = ap.len(x);
        boundltype = 0;
        boundl = 0;
        boundrtype = 0;
        boundr = 0;
        n2 = ap.len(x2);
        spline1d.spline1dconvdiff2cubic(x, y, n, boundltype, boundl, boundrtype, boundr, x2, n2, ref y2, ref d2, ref dd2);

        return;
    }

    /*************************************************************************
    This subroutine builds Catmull-Rom spline interpolant.

    INPUT PARAMETERS:
        X           -   spline nodes, array[0..N-1].
        Y           -   function values, array[0..N-1].

    OPTIONAL PARAMETERS:
        N           -   points count:
                        * N>=2
                        * if given, only first N points are used to build spline
                        * if not given, automatically detected from X/Y sizes
                          (len(X) must be equal to len(Y))
        BoundType   -   boundary condition type:
                        * -1 for periodic boundary condition
                        *  0 for parabolically terminated spline (default)
        Tension     -   tension parameter:
                        * tension=0   corresponds to classic Catmull-Rom spline (default)
                        * 0<tension<1 corresponds to more general form - cardinal spline

    OUTPUT PARAMETERS:
        C           -   spline interpolant


    ORDER OF POINTS

    Subroutine automatically sorts points, so caller may pass unsorted array.

    PROBLEMS WITH PERIODIC BOUNDARY CONDITIONS:

    Problems with periodic boundary conditions have Y[first_point]=Y[last_point].
    However, this subroutine doesn't require you to specify equal  values  for
    the first and last points - it automatically forces them  to  be  equal by
    copying  Y[first_point]  (corresponds  to the leftmost,  minimal  X[])  to
    Y[last_point]. However it is recommended to pass consistent values of Y[],
    i.e. to make Y[first_point]=Y[last_point].

      -- ALGLIB PROJECT --
         Copyright 23.06.2007 by Bochkanov Sergey
    *************************************************************************/
    public static void spline1dbuildcatmullrom(double[] x, double[] y, int n, int boundtype, double tension, out spline1dinterpolant c)
    {
        c = new spline1dinterpolant();
        spline1d.spline1dbuildcatmullrom(x, y, n, boundtype, tension, c.innerobj);
        return;
    }
    public static void spline1dbuildcatmullrom(double[] x, double[] y, out spline1dinterpolant c)
    {
        int n;
        int boundtype;
        double tension;
        if( (ap.len(x)!=ap.len(y)))
            throw new alglibexception("Error while calling 'spline1dbuildcatmullrom': looks like one of arguments has wrong size");
        c = new spline1dinterpolant();
        n = ap.len(x);
        boundtype = 0;
        tension = 0;
        spline1d.spline1dbuildcatmullrom(x, y, n, boundtype, tension, c.innerobj);

        return;
    }

    /*************************************************************************
    This subroutine builds Hermite spline interpolant.

    INPUT PARAMETERS:
        X           -   spline nodes, array[0..N-1]
        Y           -   function values, array[0..N-1]
        D           -   derivatives, array[0..N-1]
        N           -   points count (optional):
                        * N>=2
                        * if given, only first N points are used to build spline
                        * if not given, automatically detected from X/Y sizes
                          (len(X) must be equal to len(Y))

    OUTPUT PARAMETERS:
        C           -   spline interpolant.


    ORDER OF POINTS

    Subroutine automatically sorts points, so caller may pass unsorted array.

      -- ALGLIB PROJECT --
         Copyright 23.06.2007 by Bochkanov Sergey
    *************************************************************************/
    public static void spline1dbuildhermite(double[] x, double[] y, double[] d, int n, out spline1dinterpolant c)
    {
        c = new spline1dinterpolant();
        spline1d.spline1dbuildhermite(x, y, d, n, c.innerobj);
        return;
    }
    public static void spline1dbuildhermite(double[] x, double[] y, double[] d, out spline1dinterpolant c)
    {
        int n;
        if( (ap.len(x)!=ap.len(y)) || (ap.len(x)!=ap.len(d)))
            throw new alglibexception("Error while calling 'spline1dbuildhermite': looks like one of arguments has wrong size");
        c = new spline1dinterpolant();
        n = ap.len(x);
        spline1d.spline1dbuildhermite(x, y, d, n, c.innerobj);

        return;
    }

    /*************************************************************************
    This subroutine builds Akima spline interpolant

    INPUT PARAMETERS:
        X           -   spline nodes, array[0..N-1]
        Y           -   function values, array[0..N-1]
        N           -   points count (optional):
                        * N>=5
                        * if given, only first N points are used to build spline
                        * if not given, automatically detected from X/Y sizes
                          (len(X) must be equal to len(Y))

    OUTPUT PARAMETERS:
        C           -   spline interpolant


    ORDER OF POINTS

    Subroutine automatically sorts points, so caller may pass unsorted array.

      -- ALGLIB PROJECT --
         Copyright 24.06.2007 by Bochkanov Sergey
    *************************************************************************/
    public static void spline1dbuildakima(double[] x, double[] y, int n, out spline1dinterpolant c)
    {
        c = new spline1dinterpolant();
        spline1d.spline1dbuildakima(x, y, n, c.innerobj);
        return;
    }
    public static void spline1dbuildakima(double[] x, double[] y, out spline1dinterpolant c)
    {
        int n;
        if( (ap.len(x)!=ap.len(y)))
            throw new alglibexception("Error while calling 'spline1dbuildakima': looks like one of arguments has wrong size");
        c = new spline1dinterpolant();
        n = ap.len(x);
        spline1d.spline1dbuildakima(x, y, n, c.innerobj);

        return;
    }

    /*************************************************************************
    This subroutine calculates the value of the spline at the given point X.

    INPUT PARAMETERS:
        C   -   spline interpolant
        X   -   point

    Result:
        S(x)

      -- ALGLIB PROJECT --
         Copyright 23.06.2007 by Bochkanov Sergey
    *************************************************************************/
    public static double spline1dcalc(spline1dinterpolant c, double x)
    {

        double result = spline1d.spline1dcalc(c.innerobj, x);
        return result;
    }

    /*************************************************************************
    This subroutine differentiates the spline.

    INPUT PARAMETERS:
        C   -   spline interpolant.
        X   -   point

    Result:
        S   -   S(x)
        DS  -   S'(x)
        D2S -   S''(x)

      -- ALGLIB PROJECT --
         Copyright 24.06.2007 by Bochkanov Sergey
    *************************************************************************/
    public static void spline1ddiff(spline1dinterpolant c, double x, out double s, out double ds, out double d2s)
    {
        s = 0;
        ds = 0;
        d2s = 0;
        spline1d.spline1ddiff(c.innerobj, x, ref s, ref ds, ref d2s);
        return;
    }

    /*************************************************************************
    This subroutine unpacks the spline into the coefficients table.

    INPUT PARAMETERS:
        C   -   spline interpolant.
        X   -   point

    Result:
        Tbl -   coefficients table, unpacked format, array[0..N-2, 0..5].
                For I = 0...N-2:
                    Tbl[I,0] = X[i]
                    Tbl[I,1] = X[i+1]
                    Tbl[I,2] = C0
                    Tbl[I,3] = C1
                    Tbl[I,4] = C2
                    Tbl[I,5] = C3
                On [x[i], x[i+1]] spline is equals to:
                    S(x) = C0 + C1*t + C2*t^2 + C3*t^3
                    t = x-x[i]

      -- ALGLIB PROJECT --
         Copyright 29.06.2007 by Bochkanov Sergey
    *************************************************************************/
    public static void spline1dunpack(spline1dinterpolant c, out int n, out double[,] tbl)
    {
        n = 0;
        tbl = new double[0,0];
        spline1d.spline1dunpack(c.innerobj, ref n, ref tbl);
        return;
    }

    /*************************************************************************
    This subroutine performs linear transformation of the spline argument.

    INPUT PARAMETERS:
        C   -   spline interpolant.
        A, B-   transformation coefficients: x = A*t + B
    Result:
        C   -   transformed spline

      -- ALGLIB PROJECT --
         Copyright 30.06.2007 by Bochkanov Sergey
    *************************************************************************/
    public static void spline1dlintransx(spline1dinterpolant c, double a, double b)
    {

        spline1d.spline1dlintransx(c.innerobj, a, b);
        return;
    }

    /*************************************************************************
    This subroutine performs linear transformation of the spline.

    INPUT PARAMETERS:
        C   -   spline interpolant.
        A, B-   transformation coefficients: S2(x) = A*S(x) + B
    Result:
        C   -   transformed spline

      -- ALGLIB PROJECT --
         Copyright 30.06.2007 by Bochkanov Sergey
    *************************************************************************/
    public static void spline1dlintransy(spline1dinterpolant c, double a, double b)
    {

        spline1d.spline1dlintransy(c.innerobj, a, b);
        return;
    }

    /*************************************************************************
    This subroutine integrates the spline.

    INPUT PARAMETERS:
        C   -   spline interpolant.
        X   -   right bound of the integration interval [a, x],
                here 'a' denotes min(x[])
    Result:
        integral(S(t)dt,a,x)

      -- ALGLIB PROJECT --
         Copyright 23.06.2007 by Bochkanov Sergey
    *************************************************************************/
    public static double spline1dintegrate(spline1dinterpolant c, double x)
    {

        double result = spline1d.spline1dintegrate(c.innerobj, x);
        return result;
    }

}
public partial class alglib
{


    /*************************************************************************
    Polynomial fitting report:
        TaskRCond       reciprocal of task's condition number
        RMSError        RMS error
        AvgError        average error
        AvgRelError     average relative error (for non-zero Y[I])
        MaxError        maximum error
    *************************************************************************/
    public class polynomialfitreport
    {
        //
        // Public declarations
        //
        public double taskrcond { get { return _innerobj.taskrcond; } set { _innerobj.taskrcond = value; } }
        public double rmserror { get { return _innerobj.rmserror; } set { _innerobj.rmserror = value; } }
        public double avgerror { get { return _innerobj.avgerror; } set { _innerobj.avgerror = value; } }
        public double avgrelerror { get { return _innerobj.avgrelerror; } set { _innerobj.avgrelerror = value; } }
        public double maxerror { get { return _innerobj.maxerror; } set { _innerobj.maxerror = value; } }

        public polynomialfitreport()
        {
            _innerobj = new lsfit.polynomialfitreport();
        }

        //
        // Although some of declarations below are public, you should not use them
        // They are intended for internal use only
        //
        private lsfit.polynomialfitreport _innerobj;
        public lsfit.polynomialfitreport innerobj { get { return _innerobj; } }
        public polynomialfitreport(lsfit.polynomialfitreport obj)
        {
            _innerobj = obj;
        }
    }


    /*************************************************************************
    Barycentric fitting report:
        RMSError        RMS error
        AvgError        average error
        AvgRelError     average relative error (for non-zero Y[I])
        MaxError        maximum error
        TaskRCond       reciprocal of task's condition number
    *************************************************************************/
    public class barycentricfitreport
    {
        //
        // Public declarations
        //
        public double taskrcond { get { return _innerobj.taskrcond; } set { _innerobj.taskrcond = value; } }
        public int dbest { get { return _innerobj.dbest; } set { _innerobj.dbest = value; } }
        public double rmserror { get { return _innerobj.rmserror; } set { _innerobj.rmserror = value; } }
        public double avgerror { get { return _innerobj.avgerror; } set { _innerobj.avgerror = value; } }
        public double avgrelerror { get { return _innerobj.avgrelerror; } set { _innerobj.avgrelerror = value; } }
        public double maxerror { get { return _innerobj.maxerror; } set { _innerobj.maxerror = value; } }

        public barycentricfitreport()
        {
            _innerobj = new lsfit.barycentricfitreport();
        }

        //
        // Although some of declarations below are public, you should not use them
        // They are intended for internal use only
        //
        private lsfit.barycentricfitreport _innerobj;
        public lsfit.barycentricfitreport innerobj { get { return _innerobj; } }
        public barycentricfitreport(lsfit.barycentricfitreport obj)
        {
            _innerobj = obj;
        }
    }


    /*************************************************************************
    Spline fitting report:
        RMSError        RMS error
        AvgError        average error
        AvgRelError     average relative error (for non-zero Y[I])
        MaxError        maximum error

    Fields  below are  filled  by   obsolete    functions   (Spline1DFitCubic,
    Spline1DFitHermite). Modern fitting functions do NOT fill these fields:
        TaskRCond       reciprocal of task's condition number
    *************************************************************************/
    public class spline1dfitreport
    {
        //
        // Public declarations
        //
        public double taskrcond { get { return _innerobj.taskrcond; } set { _innerobj.taskrcond = value; } }
        public double rmserror { get { return _innerobj.rmserror; } set { _innerobj.rmserror = value; } }
        public double avgerror { get { return _innerobj.avgerror; } set { _innerobj.avgerror = value; } }
        public double avgrelerror { get { return _innerobj.avgrelerror; } set { _innerobj.avgrelerror = value; } }
        public double maxerror { get { return _innerobj.maxerror; } set { _innerobj.maxerror = value; } }

        public spline1dfitreport()
        {
            _innerobj = new lsfit.spline1dfitreport();
        }

        //
        // Although some of declarations below are public, you should not use them
        // They are intended for internal use only
        //
        private lsfit.spline1dfitreport _innerobj;
        public lsfit.spline1dfitreport innerobj { get { return _innerobj; } }
        public spline1dfitreport(lsfit.spline1dfitreport obj)
        {
            _innerobj = obj;
        }
    }


    /*************************************************************************
    Least squares fitting report:
        TaskRCond       reciprocal of task's condition number
        IterationsCount number of internal iterations

        RMSError        RMS error
        AvgError        average error
        AvgRelError     average relative error (for non-zero Y[I])
        MaxError        maximum error

        WRMSError       weighted RMS error
    *************************************************************************/
    public class lsfitreport
    {
        //
        // Public declarations
        //
        public double taskrcond { get { return _innerobj.taskrcond; } set { _innerobj.taskrcond = value; } }
        public int iterationscount { get { return _innerobj.iterationscount; } set { _innerobj.iterationscount = value; } }
        public double rmserror { get { return _innerobj.rmserror; } set { _innerobj.rmserror = value; } }
        public double avgerror { get { return _innerobj.avgerror; } set { _innerobj.avgerror = value; } }
        public double avgrelerror { get { return _innerobj.avgrelerror; } set { _innerobj.avgrelerror = value; } }
        public double maxerror { get { return _innerobj.maxerror; } set { _innerobj.maxerror = value; } }
        public double wrmserror { get { return _innerobj.wrmserror; } set { _innerobj.wrmserror = value; } }

        public lsfitreport()
        {
            _innerobj = new lsfit.lsfitreport();
        }

        //
        // Although some of declarations below are public, you should not use them
        // They are intended for internal use only
        //
        private lsfit.lsfitreport _innerobj;
        public lsfit.lsfitreport innerobj { get { return _innerobj; } }
        public lsfitreport(lsfit.lsfitreport obj)
        {
            _innerobj = obj;
        }
    }


    /*************************************************************************
    Nonlinear fitter.

    You should use ALGLIB functions to work with fitter.
    Never try to access its fields directly!
    *************************************************************************/
    public class lsfitstate
    {
        //
        // Public declarations
        //
        public bool needf { get { return _innerobj.needf; } set { _innerobj.needf = value; } }
        public bool needfg { get { return _innerobj.needfg; } set { _innerobj.needfg = value; } }
        public bool needfgh { get { return _innerobj.needfgh; } set { _innerobj.needfgh = value; } }
        public bool xupdated { get { return _innerobj.xupdated; } set { _innerobj.xupdated = value; } }
        public double[] c { get { return _innerobj.c; } }
        public double f { get { return _innerobj.f; } set { _innerobj.f = value; } }
        public double[] g { get { return _innerobj.g; } }
        public double[,] h { get { return _innerobj.h; } }
        public double[] x { get { return _innerobj.x; } }

        public lsfitstate()
        {
            _innerobj = new lsfit.lsfitstate();
        }

        //
        // Although some of declarations below are public, you should not use them
        // They are intended for internal use only
        //
        private lsfit.lsfitstate _innerobj;
        public lsfit.lsfitstate innerobj { get { return _innerobj; } }
        public lsfitstate(lsfit.lsfitstate obj)
        {
            _innerobj = obj;
        }
    }

    /*************************************************************************
    Fitting by polynomials in barycentric form. This function provides  simple
    unterface for unconstrained unweighted fitting. See  PolynomialFitWC()  if
    you need constrained fitting.

    Task is linear, so linear least squares solver is used. Complexity of this
    computational scheme is O(N*M^2), mostly dominated by least squares solver

    SEE ALSO:
        PolynomialFitWC()

    INPUT PARAMETERS:
        X   -   points, array[0..N-1].
        Y   -   function values, array[0..N-1].
        N   -   number of points, N>0
                * if given, only leading N elements of X/Y are used
                * if not given, automatically determined from sizes of X/Y
        M   -   number of basis functions (= polynomial_degree + 1), M>=1

    OUTPUT PARAMETERS:
        Info-   same format as in LSFitLinearW() subroutine:
                * Info>0    task is solved
                * Info<=0   an error occured:
                            -4 means inconvergence of internal SVD
        P   -   interpolant in barycentric form.
        Rep -   report, same format as in LSFitLinearW() subroutine.
                Following fields are set:
                * RMSError      rms error on the (X,Y).
                * AvgError      average error on the (X,Y).
                * AvgRelError   average relative error on the non-zero Y
                * MaxError      maximum error
                                NON-WEIGHTED ERRORS ARE CALCULATED

    NOTES:
        you can convert P from barycentric form  to  the  power  or  Chebyshev
        basis with PolynomialBar2Pow() or PolynomialBar2Cheb() functions  from
        POLINT subpackage.

      -- ALGLIB PROJECT --
         Copyright 10.12.2009 by Bochkanov Sergey
    *************************************************************************/
    public static void polynomialfit(double[] x, double[] y, int n, int m, out int info, out barycentricinterpolant p, out polynomialfitreport rep)
    {
        info = 0;
        p = new barycentricinterpolant();
        rep = new polynomialfitreport();
        lsfit.polynomialfit(x, y, n, m, ref info, p.innerobj, rep.innerobj);
        return;
    }
    public static void polynomialfit(double[] x, double[] y, int m, out int info, out barycentricinterpolant p, out polynomialfitreport rep)
    {
        int n;
        if( (ap.len(x)!=ap.len(y)))
            throw new alglibexception("Error while calling 'polynomialfit': looks like one of arguments has wrong size");
        info = 0;
        p = new barycentricinterpolant();
        rep = new polynomialfitreport();
        n = ap.len(x);
        lsfit.polynomialfit(x, y, n, m, ref info, p.innerobj, rep.innerobj);

        return;
    }

    /*************************************************************************
    Weighted  fitting by polynomials in barycentric form, with constraints  on
    function values or first derivatives.

    Small regularizing term is used when solving constrained tasks (to improve
    stability).

    Task is linear, so linear least squares solver is used. Complexity of this
    computational scheme is O(N*M^2), mostly dominated by least squares solver

    SEE ALSO:
        PolynomialFit()

    INPUT PARAMETERS:
        X   -   points, array[0..N-1].
        Y   -   function values, array[0..N-1].
        W   -   weights, array[0..N-1]
                Each summand in square  sum  of  approximation deviations from
                given  values  is  multiplied  by  the square of corresponding
                weight. Fill it by 1's if you don't  want  to  solve  weighted
                task.
        N   -   number of points, N>0.
                * if given, only leading N elements of X/Y/W are used
                * if not given, automatically determined from sizes of X/Y/W
        XC  -   points where polynomial values/derivatives are constrained,
                array[0..K-1].
        YC  -   values of constraints, array[0..K-1]
        DC  -   array[0..K-1], types of constraints:
                * DC[i]=0   means that P(XC[i])=YC[i]
                * DC[i]=1   means that P'(XC[i])=YC[i]
                SEE BELOW FOR IMPORTANT INFORMATION ON CONSTRAINTS
        K   -   number of constraints, 0<=K<M.
                K=0 means no constraints (XC/YC/DC are not used in such cases)
        M   -   number of basis functions (= polynomial_degree + 1), M>=1

    OUTPUT PARAMETERS:
        Info-   same format as in LSFitLinearW() subroutine:
                * Info>0    task is solved
                * Info<=0   an error occured:
                            -4 means inconvergence of internal SVD
                            -3 means inconsistent constraints
        P   -   interpolant in barycentric form.
        Rep -   report, same format as in LSFitLinearW() subroutine.
                Following fields are set:
                * RMSError      rms error on the (X,Y).
                * AvgError      average error on the (X,Y).
                * AvgRelError   average relative error on the non-zero Y
                * MaxError      maximum error
                                NON-WEIGHTED ERRORS ARE CALCULATED

    IMPORTANT:
        this subroitine doesn't calculate task's condition number for K<>0.

    NOTES:
        you can convert P from barycentric form  to  the  power  or  Chebyshev
        basis with PolynomialBar2Pow() or PolynomialBar2Cheb() functions  from
        POLINT subpackage.

    SETTING CONSTRAINTS - DANGERS AND OPPORTUNITIES:

    Setting constraints can lead  to undesired  results,  like ill-conditioned
    behavior, or inconsistency being detected. From the other side,  it allows
    us to improve quality of the fit. Here we summarize  our  experience  with
    constrained regression splines:
    * even simple constraints can be inconsistent, see  Wikipedia  article  on
      this subject: http://en.wikipedia.org/wiki/Birkhoff_interpolation
    * the  greater  is  M (given  fixed  constraints),  the  more chances that
      constraints will be consistent
    * in the general case, consistency of constraints is NOT GUARANTEED.
    * in the one special cases, however, we can  guarantee  consistency.  This
      case  is:  M>1  and constraints on the function values (NOT DERIVATIVES)

    Our final recommendation is to use constraints  WHEN  AND  ONLY  when  you
    can't solve your task without them. Anything beyond  special  cases  given
    above is not guaranteed and may result in inconsistency.

      -- ALGLIB PROJECT --
         Copyright 10.12.2009 by Bochkanov Sergey
    *************************************************************************/
    public static void polynomialfitwc(double[] x, double[] y, double[] w, int n, double[] xc, double[] yc, int[] dc, int k, int m, out int info, out barycentricinterpolant p, out polynomialfitreport rep)
    {
        info = 0;
        p = new barycentricinterpolant();
        rep = new polynomialfitreport();
        lsfit.polynomialfitwc(x, y, w, n, xc, yc, dc, k, m, ref info, p.innerobj, rep.innerobj);
        return;
    }
    public static void polynomialfitwc(double[] x, double[] y, double[] w, double[] xc, double[] yc, int[] dc, int m, out int info, out barycentricinterpolant p, out polynomialfitreport rep)
    {
        int n;
        int k;
        if( (ap.len(x)!=ap.len(y)) || (ap.len(x)!=ap.len(w)))
            throw new alglibexception("Error while calling 'polynomialfitwc': looks like one of arguments has wrong size");
        if( (ap.len(xc)!=ap.len(yc)) || (ap.len(xc)!=ap.len(dc)))
            throw new alglibexception("Error while calling 'polynomialfitwc': looks like one of arguments has wrong size");
        info = 0;
        p = new barycentricinterpolant();
        rep = new polynomialfitreport();
        n = ap.len(x);
        k = ap.len(xc);
        lsfit.polynomialfitwc(x, y, w, n, xc, yc, dc, k, m, ref info, p.innerobj, rep.innerobj);

        return;
    }

    /*************************************************************************
    Weghted rational least  squares  fitting  using  Floater-Hormann  rational
    functions  with  optimal  D  chosen  from  [0,9],  with  constraints   and
    individual weights.

    Equidistant  grid  with M node on [min(x),max(x)]  is  used to build basis
    functions. Different values of D are tried, optimal D (least WEIGHTED root
    mean square error) is chosen.  Task  is  linear,  so  linear least squares
    solver  is  used.  Complexity  of  this  computational  scheme is O(N*M^2)
    (mostly dominated by the least squares solver).

    SEE ALSO
    * BarycentricFitFloaterHormann(), "lightweight" fitting without invididual
      weights and constraints.

    INPUT PARAMETERS:
        X   -   points, array[0..N-1].
        Y   -   function values, array[0..N-1].
        W   -   weights, array[0..N-1]
                Each summand in square  sum  of  approximation deviations from
                given  values  is  multiplied  by  the square of corresponding
                weight. Fill it by 1's if you don't  want  to  solve  weighted
                task.
        N   -   number of points, N>0.
        XC  -   points where function values/derivatives are constrained,
                array[0..K-1].
        YC  -   values of constraints, array[0..K-1]
        DC  -   array[0..K-1], types of constraints:
                * DC[i]=0   means that S(XC[i])=YC[i]
                * DC[i]=1   means that S'(XC[i])=YC[i]
                SEE BELOW FOR IMPORTANT INFORMATION ON CONSTRAINTS
        K   -   number of constraints, 0<=K<M.
                K=0 means no constraints (XC/YC/DC are not used in such cases)
        M   -   number of basis functions ( = number_of_nodes), M>=2.

    OUTPUT PARAMETERS:
        Info-   same format as in LSFitLinearWC() subroutine.
                * Info>0    task is solved
                * Info<=0   an error occured:
                            -4 means inconvergence of internal SVD
                            -3 means inconsistent constraints
                            -1 means another errors in parameters passed
                               (N<=0, for example)
        B   -   barycentric interpolant.
        Rep -   report, same format as in LSFitLinearWC() subroutine.
                Following fields are set:
                * DBest         best value of the D parameter
                * RMSError      rms error on the (X,Y).
                * AvgError      average error on the (X,Y).
                * AvgRelError   average relative error on the non-zero Y
                * MaxError      maximum error
                                NON-WEIGHTED ERRORS ARE CALCULATED

    IMPORTANT:
        this subroutine doesn't calculate task's condition number for K<>0.

    SETTING CONSTRAINTS - DANGERS AND OPPORTUNITIES:

    Setting constraints can lead  to undesired  results,  like ill-conditioned
    behavior, or inconsistency being detected. From the other side,  it allows
    us to improve quality of the fit. Here we summarize  our  experience  with
    constrained barycentric interpolants:
    * excessive  constraints  can  be  inconsistent.   Floater-Hormann   basis
      functions aren't as flexible as splines (although they are very smooth).
    * the more evenly constraints are spread across [min(x),max(x)],  the more
      chances that they will be consistent
    * the  greater  is  M (given  fixed  constraints),  the  more chances that
      constraints will be consistent
    * in the general case, consistency of constraints IS NOT GUARANTEED.
    * in the several special cases, however, we CAN guarantee consistency.
    * one of this cases is constraints on the function  VALUES at the interval
      boundaries. Note that consustency of the  constraints  on  the  function
      DERIVATIVES is NOT guaranteed (you can use in such cases  cubic  splines
      which are more flexible).
    * another  special  case  is ONE constraint on the function value (OR, but
      not AND, derivative) anywhere in the interval

    Our final recommendation is to use constraints  WHEN  AND  ONLY  WHEN  you
    can't solve your task without them. Anything beyond  special  cases  given
    above is not guaranteed and may result in inconsistency.

      -- ALGLIB PROJECT --
         Copyright 18.08.2009 by Bochkanov Sergey
    *************************************************************************/
    public static void barycentricfitfloaterhormannwc(double[] x, double[] y, double[] w, int n, double[] xc, double[] yc, int[] dc, int k, int m, out int info, out barycentricinterpolant b, out barycentricfitreport rep)
    {
        info = 0;
        b = new barycentricinterpolant();
        rep = new barycentricfitreport();
        lsfit.barycentricfitfloaterhormannwc(x, y, w, n, xc, yc, dc, k, m, ref info, b.innerobj, rep.innerobj);
        return;
    }

    /*************************************************************************
    Rational least squares fitting using  Floater-Hormann  rational  functions
    with optimal D chosen from [0,9].

    Equidistant  grid  with M node on [min(x),max(x)]  is  used to build basis
    functions. Different values of D are tried, optimal  D  (least  root  mean
    square error) is chosen.  Task  is  linear, so linear least squares solver
    is used. Complexity  of  this  computational  scheme is  O(N*M^2)  (mostly
    dominated by the least squares solver).

    INPUT PARAMETERS:
        X   -   points, array[0..N-1].
        Y   -   function values, array[0..N-1].
        N   -   number of points, N>0.
        M   -   number of basis functions ( = number_of_nodes), M>=2.

    OUTPUT PARAMETERS:
        Info-   same format as in LSFitLinearWC() subroutine.
                * Info>0    task is solved
                * Info<=0   an error occured:
                            -4 means inconvergence of internal SVD
                            -3 means inconsistent constraints
        B   -   barycentric interpolant.
        Rep -   report, same format as in LSFitLinearWC() subroutine.
                Following fields are set:
                * DBest         best value of the D parameter
                * RMSError      rms error on the (X,Y).
                * AvgError      average error on the (X,Y).
                * AvgRelError   average relative error on the non-zero Y
                * MaxError      maximum error
                                NON-WEIGHTED ERRORS ARE CALCULATED

      -- ALGLIB PROJECT --
         Copyright 18.08.2009 by Bochkanov Sergey
    *************************************************************************/
    public static void barycentricfitfloaterhormann(double[] x, double[] y, int n, int m, out int info, out barycentricinterpolant b, out barycentricfitreport rep)
    {
        info = 0;
        b = new barycentricinterpolant();
        rep = new barycentricfitreport();
        lsfit.barycentricfitfloaterhormann(x, y, n, m, ref info, b.innerobj, rep.innerobj);
        return;
    }

    /*************************************************************************
    Rational least squares fitting using  Floater-Hormann  rational  functions
    with optimal D chosen from [0,9].

    Equidistant  grid  with M node on [min(x),max(x)]  is  used to build basis
    functions. Different values of D are tried, optimal  D  (least  root  mean
    square error) is chosen.  Task  is  linear, so linear least squares solver
    is used. Complexity  of  this  computational  scheme is  O(N*M^2)  (mostly
    dominated by the least squares solver).

    INPUT PARAMETERS:
        X   -   points, array[0..N-1].
        Y   -   function values, array[0..N-1].
        N   -   number of points, N>0.
        M   -   number of basis functions ( = number_of_nodes), M>=2.

    OUTPUT PARAMETERS:
        Info-   same format as in LSFitLinearWC() subroutine.
                * Info>0    task is solved
                * Info<=0   an error occured:
                            -4 means inconvergence of internal SVD
                            -3 means inconsistent constraints
        B   -   barycentric interpolant.
        Rep -   report, same format as in LSFitLinearWC() subroutine.
                Following fields are set:
                * DBest         best value of the D parameter
                * RMSError      rms error on the (X,Y).
                * AvgError      average error on the (X,Y).
                * AvgRelError   average relative error on the non-zero Y
                * MaxError      maximum error
                                NON-WEIGHTED ERRORS ARE CALCULATED

      -- ALGLIB PROJECT --
         Copyright 18.08.2009 by Bochkanov Sergey
    *************************************************************************/
    public static void spline1dfitpenalized(double[] x, double[] y, int n, int m, double rho, out int info, out spline1dinterpolant s, out spline1dfitreport rep)
    {
        info = 0;
        s = new spline1dinterpolant();
        rep = new spline1dfitreport();
        lsfit.spline1dfitpenalized(x, y, n, m, rho, ref info, s.innerobj, rep.innerobj);
        return;
    }
    public static void spline1dfitpenalized(double[] x, double[] y, int m, double rho, out int info, out spline1dinterpolant s, out spline1dfitreport rep)
    {
        int n;
        if( (ap.len(x)!=ap.len(y)))
            throw new alglibexception("Error while calling 'spline1dfitpenalized': looks like one of arguments has wrong size");
        info = 0;
        s = new spline1dinterpolant();
        rep = new spline1dfitreport();
        n = ap.len(x);
        lsfit.spline1dfitpenalized(x, y, n, m, rho, ref info, s.innerobj, rep.innerobj);

        return;
    }

    /*************************************************************************
    Weighted fitting by penalized cubic spline.

    Equidistant grid with M nodes on [min(x,xc),max(x,xc)] is  used  to  build
    basis functions. Basis functions are cubic splines with  natural  boundary
    conditions. Problem is regularized by  adding non-linearity penalty to the
    usual least squares penalty function:

        S(x) = arg min { LS + P }, where
        LS   = SUM { w[i]^2*(y[i] - S(x[i]))^2 } - least squares penalty
        P    = C*10^rho*integral{ S''(x)^2*dx } - non-linearity penalty
        rho  - tunable constant given by user
        C    - automatically determined scale parameter,
               makes penalty invariant with respect to scaling of X, Y, W.

    INPUT PARAMETERS:
        X   -   points, array[0..N-1].
        Y   -   function values, array[0..N-1].
        W   -   weights, array[0..N-1]
                Each summand in square  sum  of  approximation deviations from
                given  values  is  multiplied  by  the square of corresponding
                weight. Fill it by 1's if you don't  want  to  solve  weighted
                problem.
        N   -   number of points (optional):
                * N>0
                * if given, only first N elements of X/Y/W are processed
                * if not given, automatically determined from X/Y/W sizes
        M   -   number of basis functions ( = number_of_nodes), M>=4.
        Rho -   regularization  constant  passed   by   user.   It   penalizes
                nonlinearity in the regression spline. It  is  logarithmically
                scaled,  i.e.  actual  value  of  regularization  constant  is
                calculated as 10^Rho. It is automatically scaled so that:
                * Rho=2.0 corresponds to moderate amount of nonlinearity
                * generally, it should be somewhere in the [-8.0,+8.0]
                If you do not want to penalize nonlineary,
                pass small Rho. Values as low as -15 should work.

    OUTPUT PARAMETERS:
        Info-   same format as in LSFitLinearWC() subroutine.
                * Info>0    task is solved
                * Info<=0   an error occured:
                            -4 means inconvergence of internal SVD or
                               Cholesky decomposition; problem may be
                               too ill-conditioned (very rare)
        S   -   spline interpolant.
        Rep -   Following fields are set:
                * RMSError      rms error on the (X,Y).
                * AvgError      average error on the (X,Y).
                * AvgRelError   average relative error on the non-zero Y
                * MaxError      maximum error
                                NON-WEIGHTED ERRORS ARE CALCULATED

    IMPORTANT:
        this subroitine doesn't calculate task's condition number for K<>0.

    NOTE 1: additional nodes are added to the spline outside  of  the  fitting
    interval to force linearity when x<min(x,xc) or x>max(x,xc).  It  is  done
    for consistency - we penalize non-linearity  at [min(x,xc),max(x,xc)],  so
    it is natural to force linearity outside of this interval.

    NOTE 2: function automatically sorts points,  so  caller may pass unsorted
    array.

      -- ALGLIB PROJECT --
         Copyright 19.10.2010 by Bochkanov Sergey
    *************************************************************************/
    public static void spline1dfitpenalizedw(double[] x, double[] y, double[] w, int n, int m, double rho, out int info, out spline1dinterpolant s, out spline1dfitreport rep)
    {
        info = 0;
        s = new spline1dinterpolant();
        rep = new spline1dfitreport();
        lsfit.spline1dfitpenalizedw(x, y, w, n, m, rho, ref info, s.innerobj, rep.innerobj);
        return;
    }
    public static void spline1dfitpenalizedw(double[] x, double[] y, double[] w, int m, double rho, out int info, out spline1dinterpolant s, out spline1dfitreport rep)
    {
        int n;
        if( (ap.len(x)!=ap.len(y)) || (ap.len(x)!=ap.len(w)))
            throw new alglibexception("Error while calling 'spline1dfitpenalizedw': looks like one of arguments has wrong size");
        info = 0;
        s = new spline1dinterpolant();
        rep = new spline1dfitreport();
        n = ap.len(x);
        lsfit.spline1dfitpenalizedw(x, y, w, n, m, rho, ref info, s.innerobj, rep.innerobj);

        return;
    }

    /*************************************************************************
    Weighted fitting by cubic  spline,  with constraints on function values or
    derivatives.

    Equidistant grid with M-2 nodes on [min(x,xc),max(x,xc)] is  used to build
    basis functions. Basis functions are cubic splines with continuous  second
    derivatives  and  non-fixed first  derivatives  at  interval  ends.  Small
    regularizing term is used  when  solving  constrained  tasks  (to  improve
    stability).

    Task is linear, so linear least squares solver is used. Complexity of this
    computational scheme is O(N*M^2), mostly dominated by least squares solver

    SEE ALSO
        Spline1DFitHermiteWC()  -   fitting by Hermite splines (more flexible,
                                    less smooth)
        Spline1DFitCubic()      -   "lightweight" fitting  by  cubic  splines,
                                    without invididual weights and constraints

    INPUT PARAMETERS:
        X   -   points, array[0..N-1].
        Y   -   function values, array[0..N-1].
        W   -   weights, array[0..N-1]
                Each summand in square  sum  of  approximation deviations from
                given  values  is  multiplied  by  the square of corresponding
                weight. Fill it by 1's if you don't  want  to  solve  weighted
                task.
        N   -   number of points (optional):
                * N>0
                * if given, only first N elements of X/Y/W are processed
                * if not given, automatically determined from X/Y/W sizes
        XC  -   points where spline values/derivatives are constrained,
                array[0..K-1].
        YC  -   values of constraints, array[0..K-1]
        DC  -   array[0..K-1], types of constraints:
                * DC[i]=0   means that S(XC[i])=YC[i]
                * DC[i]=1   means that S'(XC[i])=YC[i]
                SEE BELOW FOR IMPORTANT INFORMATION ON CONSTRAINTS
        K   -   number of constraints (optional):
                * 0<=K<M.
                * K=0 means no constraints (XC/YC/DC are not used)
                * if given, only first K elements of XC/YC/DC are used
                * if not given, automatically determined from XC/YC/DC
        M   -   number of basis functions ( = number_of_nodes+2), M>=4.

    OUTPUT PARAMETERS:
        Info-   same format as in LSFitLinearWC() subroutine.
                * Info>0    task is solved
                * Info<=0   an error occured:
                            -4 means inconvergence of internal SVD
                            -3 means inconsistent constraints
        S   -   spline interpolant.
        Rep -   report, same format as in LSFitLinearWC() subroutine.
                Following fields are set:
                * RMSError      rms error on the (X,Y).
                * AvgError      average error on the (X,Y).
                * AvgRelError   average relative error on the non-zero Y
                * MaxError      maximum error
                                NON-WEIGHTED ERRORS ARE CALCULATED

    IMPORTANT:
        this subroitine doesn't calculate task's condition number for K<>0.


    ORDER OF POINTS

    Subroutine automatically sorts points, so caller may pass unsorted array.

    SETTING CONSTRAINTS - DANGERS AND OPPORTUNITIES:

    Setting constraints can lead  to undesired  results,  like ill-conditioned
    behavior, or inconsistency being detected. From the other side,  it allows
    us to improve quality of the fit. Here we summarize  our  experience  with
    constrained regression splines:
    * excessive constraints can be inconsistent. Splines are  piecewise  cubic
      functions, and it is easy to create an example, where  large  number  of
      constraints  concentrated  in  small  area will result in inconsistency.
      Just because spline is not flexible enough to satisfy all of  them.  And
      same constraints spread across the  [min(x),max(x)]  will  be  perfectly
      consistent.
    * the more evenly constraints are spread across [min(x),max(x)],  the more
      chances that they will be consistent
    * the  greater  is  M (given  fixed  constraints),  the  more chances that
      constraints will be consistent
    * in the general case, consistency of constraints IS NOT GUARANTEED.
    * in the several special cases, however, we CAN guarantee consistency.
    * one of this cases is constraints  on  the  function  values  AND/OR  its
      derivatives at the interval boundaries.
    * another  special  case  is ONE constraint on the function value (OR, but
      not AND, derivative) anywhere in the interval

    Our final recommendation is to use constraints  WHEN  AND  ONLY  WHEN  you
    can't solve your task without them. Anything beyond  special  cases  given
    above is not guaranteed and may result in inconsistency.


      -- ALGLIB PROJECT --
         Copyright 18.08.2009 by Bochkanov Sergey
    *************************************************************************/
    public static void spline1dfitcubicwc(double[] x, double[] y, double[] w, int n, double[] xc, double[] yc, int[] dc, int k, int m, out int info, out spline1dinterpolant s, out spline1dfitreport rep)
    {
        info = 0;
        s = new spline1dinterpolant();
        rep = new spline1dfitreport();
        lsfit.spline1dfitcubicwc(x, y, w, n, xc, yc, dc, k, m, ref info, s.innerobj, rep.innerobj);
        return;
    }
    public static void spline1dfitcubicwc(double[] x, double[] y, double[] w, double[] xc, double[] yc, int[] dc, int m, out int info, out spline1dinterpolant s, out spline1dfitreport rep)
    {
        int n;
        int k;
        if( (ap.len(x)!=ap.len(y)) || (ap.len(x)!=ap.len(w)))
            throw new alglibexception("Error while calling 'spline1dfitcubicwc': looks like one of arguments has wrong size");
        if( (ap.len(xc)!=ap.len(yc)) || (ap.len(xc)!=ap.len(dc)))
            throw new alglibexception("Error while calling 'spline1dfitcubicwc': looks like one of arguments has wrong size");
        info = 0;
        s = new spline1dinterpolant();
        rep = new spline1dfitreport();
        n = ap.len(x);
        k = ap.len(xc);
        lsfit.spline1dfitcubicwc(x, y, w, n, xc, yc, dc, k, m, ref info, s.innerobj, rep.innerobj);

        return;
    }

    /*************************************************************************
    Weighted  fitting  by Hermite spline,  with constraints on function values
    or first derivatives.

    Equidistant grid with M nodes on [min(x,xc),max(x,xc)] is  used  to  build
    basis functions. Basis functions are Hermite splines.  Small  regularizing
    term is used when solving constrained tasks (to improve stability).

    Task is linear, so linear least squares solver is used. Complexity of this
    computational scheme is O(N*M^2), mostly dominated by least squares solver

    SEE ALSO
        Spline1DFitCubicWC()    -   fitting by Cubic splines (less flexible,
                                    more smooth)
        Spline1DFitHermite()    -   "lightweight" Hermite fitting, without
                                    invididual weights and constraints

    INPUT PARAMETERS:
        X   -   points, array[0..N-1].
        Y   -   function values, array[0..N-1].
        W   -   weights, array[0..N-1]
                Each summand in square  sum  of  approximation deviations from
                given  values  is  multiplied  by  the square of corresponding
                weight. Fill it by 1's if you don't  want  to  solve  weighted
                task.
        N   -   number of points (optional):
                * N>0
                * if given, only first N elements of X/Y/W are processed
                * if not given, automatically determined from X/Y/W sizes
        XC  -   points where spline values/derivatives are constrained,
                array[0..K-1].
        YC  -   values of constraints, array[0..K-1]
        DC  -   array[0..K-1], types of constraints:
                * DC[i]=0   means that S(XC[i])=YC[i]
                * DC[i]=1   means that S'(XC[i])=YC[i]
                SEE BELOW FOR IMPORTANT INFORMATION ON CONSTRAINTS
        K   -   number of constraints (optional):
                * 0<=K<M.
                * K=0 means no constraints (XC/YC/DC are not used)
                * if given, only first K elements of XC/YC/DC are used
                * if not given, automatically determined from XC/YC/DC
        M   -   number of basis functions (= 2 * number of nodes),
                M>=4,
                M IS EVEN!

    OUTPUT PARAMETERS:
        Info-   same format as in LSFitLinearW() subroutine:
                * Info>0    task is solved
                * Info<=0   an error occured:
                            -4 means inconvergence of internal SVD
                            -3 means inconsistent constraints
                            -2 means odd M was passed (which is not supported)
                            -1 means another errors in parameters passed
                               (N<=0, for example)
        S   -   spline interpolant.
        Rep -   report, same format as in LSFitLinearW() subroutine.
                Following fields are set:
                * RMSError      rms error on the (X,Y).
                * AvgError      average error on the (X,Y).
                * AvgRelError   average relative error on the non-zero Y
                * MaxError      maximum error
                                NON-WEIGHTED ERRORS ARE CALCULATED

    IMPORTANT:
        this subroitine doesn't calculate task's condition number for K<>0.

    IMPORTANT:
        this subroitine supports only even M's


    ORDER OF POINTS

    Subroutine automatically sorts points, so caller may pass unsorted array.

    SETTING CONSTRAINTS - DANGERS AND OPPORTUNITIES:

    Setting constraints can lead  to undesired  results,  like ill-conditioned
    behavior, or inconsistency being detected. From the other side,  it allows
    us to improve quality of the fit. Here we summarize  our  experience  with
    constrained regression splines:
    * excessive constraints can be inconsistent. Splines are  piecewise  cubic
      functions, and it is easy to create an example, where  large  number  of
      constraints  concentrated  in  small  area will result in inconsistency.
      Just because spline is not flexible enough to satisfy all of  them.  And
      same constraints spread across the  [min(x),max(x)]  will  be  perfectly
      consistent.
    * the more evenly constraints are spread across [min(x),max(x)],  the more
      chances that they will be consistent
    * the  greater  is  M (given  fixed  constraints),  the  more chances that
      constraints will be consistent
    * in the general case, consistency of constraints is NOT GUARANTEED.
    * in the several special cases, however, we can guarantee consistency.
    * one of this cases is  M>=4  and   constraints  on   the  function  value
      (AND/OR its derivative) at the interval boundaries.
    * another special case is M>=4  and  ONE  constraint on the function value
      (OR, BUT NOT AND, derivative) anywhere in [min(x),max(x)]

    Our final recommendation is to use constraints  WHEN  AND  ONLY  when  you
    can't solve your task without them. Anything beyond  special  cases  given
    above is not guaranteed and may result in inconsistency.

      -- ALGLIB PROJECT --
         Copyright 18.08.2009 by Bochkanov Sergey
    *************************************************************************/
    public static void spline1dfithermitewc(double[] x, double[] y, double[] w, int n, double[] xc, double[] yc, int[] dc, int k, int m, out int info, out spline1dinterpolant s, out spline1dfitreport rep)
    {
        info = 0;
        s = new spline1dinterpolant();
        rep = new spline1dfitreport();
        lsfit.spline1dfithermitewc(x, y, w, n, xc, yc, dc, k, m, ref info, s.innerobj, rep.innerobj);
        return;
    }
    public static void spline1dfithermitewc(double[] x, double[] y, double[] w, double[] xc, double[] yc, int[] dc, int m, out int info, out spline1dinterpolant s, out spline1dfitreport rep)
    {
        int n;
        int k;
        if( (ap.len(x)!=ap.len(y)) || (ap.len(x)!=ap.len(w)))
            throw new alglibexception("Error while calling 'spline1dfithermitewc': looks like one of arguments has wrong size");
        if( (ap.len(xc)!=ap.len(yc)) || (ap.len(xc)!=ap.len(dc)))
            throw new alglibexception("Error while calling 'spline1dfithermitewc': looks like one of arguments has wrong size");
        info = 0;
        s = new spline1dinterpolant();
        rep = new spline1dfitreport();
        n = ap.len(x);
        k = ap.len(xc);
        lsfit.spline1dfithermitewc(x, y, w, n, xc, yc, dc, k, m, ref info, s.innerobj, rep.innerobj);

        return;
    }

    /*************************************************************************
    Least squares fitting by cubic spline.

    This subroutine is "lightweight" alternative for more complex and feature-
    rich Spline1DFitCubicWC().  See  Spline1DFitCubicWC() for more information
    about subroutine parameters (we don't duplicate it here because of length)

      -- ALGLIB PROJECT --
         Copyright 18.08.2009 by Bochkanov Sergey
    *************************************************************************/
    public static void spline1dfitcubic(double[] x, double[] y, int n, int m, out int info, out spline1dinterpolant s, out spline1dfitreport rep)
    {
        info = 0;
        s = new spline1dinterpolant();
        rep = new spline1dfitreport();
        lsfit.spline1dfitcubic(x, y, n, m, ref info, s.innerobj, rep.innerobj);
        return;
    }
    public static void spline1dfitcubic(double[] x, double[] y, int m, out int info, out spline1dinterpolant s, out spline1dfitreport rep)
    {
        int n;
        if( (ap.len(x)!=ap.len(y)))
            throw new alglibexception("Error while calling 'spline1dfitcubic': looks like one of arguments has wrong size");
        info = 0;
        s = new spline1dinterpolant();
        rep = new spline1dfitreport();
        n = ap.len(x);
        lsfit.spline1dfitcubic(x, y, n, m, ref info, s.innerobj, rep.innerobj);

        return;
    }

    /*************************************************************************
    Least squares fitting by Hermite spline.

    This subroutine is "lightweight" alternative for more complex and feature-
    rich Spline1DFitHermiteWC().  See Spline1DFitHermiteWC()  description  for
    more information about subroutine parameters (we don't duplicate  it  here
    because of length).

      -- ALGLIB PROJECT --
         Copyright 18.08.2009 by Bochkanov Sergey
    *************************************************************************/
    public static void spline1dfithermite(double[] x, double[] y, int n, int m, out int info, out spline1dinterpolant s, out spline1dfitreport rep)
    {
        info = 0;
        s = new spline1dinterpolant();
        rep = new spline1dfitreport();
        lsfit.spline1dfithermite(x, y, n, m, ref info, s.innerobj, rep.innerobj);
        return;
    }
    public static void spline1dfithermite(double[] x, double[] y, int m, out int info, out spline1dinterpolant s, out spline1dfitreport rep)
    {
        int n;
        if( (ap.len(x)!=ap.len(y)))
            throw new alglibexception("Error while calling 'spline1dfithermite': looks like one of arguments has wrong size");
        info = 0;
        s = new spline1dinterpolant();
        rep = new spline1dfitreport();
        n = ap.len(x);
        lsfit.spline1dfithermite(x, y, n, m, ref info, s.innerobj, rep.innerobj);

        return;
    }

    /*************************************************************************
    Weighted linear least squares fitting.

    QR decomposition is used to reduce task to MxM, then triangular solver  or
    SVD-based solver is used depending on condition number of the  system.  It
    allows to maximize speed and retain decent accuracy.

    INPUT PARAMETERS:
        Y       -   array[0..N-1] Function values in  N  points.
        W       -   array[0..N-1]  Weights  corresponding to function  values.
                    Each summand in square  sum  of  approximation  deviations
                    from  given  values  is  multiplied  by  the   square   of
                    corresponding weight.
        FMatrix -   a table of basis functions values, array[0..N-1, 0..M-1].
                    FMatrix[I, J] - value of J-th basis function in I-th point.
        N       -   number of points used. N>=1.
        M       -   number of basis functions, M>=1.

    OUTPUT PARAMETERS:
        Info    -   error code:
                    * -4    internal SVD decomposition subroutine failed (very
                            rare and for degenerate systems only)
                    * -1    incorrect N/M were specified
                    *  1    task is solved
        C       -   decomposition coefficients, array[0..M-1]
        Rep     -   fitting report. Following fields are set:
                    * Rep.TaskRCond     reciprocal of condition number
                    * RMSError          rms error on the (X,Y).
                    * AvgError          average error on the (X,Y).
                    * AvgRelError       average relative error on the non-zero Y
                    * MaxError          maximum error
                                        NON-WEIGHTED ERRORS ARE CALCULATED

      -- ALGLIB --
         Copyright 17.08.2009 by Bochkanov Sergey
    *************************************************************************/
    public static void lsfitlinearw(double[] y, double[] w, double[,] fmatrix, int n, int m, out int info, out double[] c, out lsfitreport rep)
    {
        info = 0;
        c = new double[0];
        rep = new lsfitreport();
        lsfit.lsfitlinearw(y, w, fmatrix, n, m, ref info, ref c, rep.innerobj);
        return;
    }
    public static void lsfitlinearw(double[] y, double[] w, double[,] fmatrix, out int info, out double[] c, out lsfitreport rep)
    {
        int n;
        int m;
        if( (ap.len(y)!=ap.len(w)) || (ap.len(y)!=ap.rows(fmatrix)))
            throw new alglibexception("Error while calling 'lsfitlinearw': looks like one of arguments has wrong size");
        info = 0;
        c = new double[0];
        rep = new lsfitreport();
        n = ap.len(y);
        m = ap.cols(fmatrix);
        lsfit.lsfitlinearw(y, w, fmatrix, n, m, ref info, ref c, rep.innerobj);

        return;
    }

    /*************************************************************************
    Weighted constained linear least squares fitting.

    This  is  variation  of LSFitLinearW(), which searchs for min|A*x=b| given
    that  K  additional  constaints  C*x=bc are satisfied. It reduces original
    task to modified one: min|B*y-d| WITHOUT constraints,  then LSFitLinearW()
    is called.

    INPUT PARAMETERS:
        Y       -   array[0..N-1] Function values in  N  points.
        W       -   array[0..N-1]  Weights  corresponding to function  values.
                    Each summand in square  sum  of  approximation  deviations
                    from  given  values  is  multiplied  by  the   square   of
                    corresponding weight.
        FMatrix -   a table of basis functions values, array[0..N-1, 0..M-1].
                    FMatrix[I,J] - value of J-th basis function in I-th point.
        CMatrix -   a table of constaints, array[0..K-1,0..M].
                    I-th row of CMatrix corresponds to I-th linear constraint:
                    CMatrix[I,0]*C[0] + ... + CMatrix[I,M-1]*C[M-1] = CMatrix[I,M]
        N       -   number of points used. N>=1.
        M       -   number of basis functions, M>=1.
        K       -   number of constraints, 0 <= K < M
                    K=0 corresponds to absence of constraints.

    OUTPUT PARAMETERS:
        Info    -   error code:
                    * -4    internal SVD decomposition subroutine failed (very
                            rare and for degenerate systems only)
                    * -3    either   too   many  constraints  (M   or   more),
                            degenerate  constraints   (some   constraints  are
                            repetead twice) or inconsistent  constraints  were
                            specified.
                    *  1    task is solved
        C       -   decomposition coefficients, array[0..M-1]
        Rep     -   fitting report. Following fields are set:
                    * RMSError          rms error on the (X,Y).
                    * AvgError          average error on the (X,Y).
                    * AvgRelError       average relative error on the non-zero Y
                    * MaxError          maximum error
                                        NON-WEIGHTED ERRORS ARE CALCULATED

    IMPORTANT:
        this subroitine doesn't calculate task's condition number for K<>0.

      -- ALGLIB --
         Copyright 07.09.2009 by Bochkanov Sergey
    *************************************************************************/
    public static void lsfitlinearwc(double[] y, double[] w, double[,] fmatrix, double[,] cmatrix, int n, int m, int k, out int info, out double[] c, out lsfitreport rep)
    {
        info = 0;
        c = new double[0];
        rep = new lsfitreport();
        lsfit.lsfitlinearwc(y, w, fmatrix, cmatrix, n, m, k, ref info, ref c, rep.innerobj);
        return;
    }
    public static void lsfitlinearwc(double[] y, double[] w, double[,] fmatrix, double[,] cmatrix, out int info, out double[] c, out lsfitreport rep)
    {
        int n;
        int m;
        int k;
        if( (ap.len(y)!=ap.len(w)) || (ap.len(y)!=ap.rows(fmatrix)))
            throw new alglibexception("Error while calling 'lsfitlinearwc': looks like one of arguments has wrong size");
        if( (ap.cols(fmatrix)!=ap.cols(cmatrix)-1))
            throw new alglibexception("Error while calling 'lsfitlinearwc': looks like one of arguments has wrong size");
        info = 0;
        c = new double[0];
        rep = new lsfitreport();
        n = ap.len(y);
        m = ap.cols(fmatrix);
        k = ap.rows(cmatrix);
        lsfit.lsfitlinearwc(y, w, fmatrix, cmatrix, n, m, k, ref info, ref c, rep.innerobj);

        return;
    }

    /*************************************************************************
    Linear least squares fitting.

    QR decomposition is used to reduce task to MxM, then triangular solver  or
    SVD-based solver is used depending on condition number of the  system.  It
    allows to maximize speed and retain decent accuracy.

    INPUT PARAMETERS:
        Y       -   array[0..N-1] Function values in  N  points.
        FMatrix -   a table of basis functions values, array[0..N-1, 0..M-1].
                    FMatrix[I, J] - value of J-th basis function in I-th point.
        N       -   number of points used. N>=1.
        M       -   number of basis functions, M>=1.

    OUTPUT PARAMETERS:
        Info    -   error code:
                    * -4    internal SVD decomposition subroutine failed (very
                            rare and for degenerate systems only)
                    *  1    task is solved
        C       -   decomposition coefficients, array[0..M-1]
        Rep     -   fitting report. Following fields are set:
                    * Rep.TaskRCond     reciprocal of condition number
                    * RMSError          rms error on the (X,Y).
                    * AvgError          average error on the (X,Y).
                    * AvgRelError       average relative error on the non-zero Y
                    * MaxError          maximum error
                                        NON-WEIGHTED ERRORS ARE CALCULATED

      -- ALGLIB --
         Copyright 17.08.2009 by Bochkanov Sergey
    *************************************************************************/
    public static void lsfitlinear(double[] y, double[,] fmatrix, int n, int m, out int info, out double[] c, out lsfitreport rep)
    {
        info = 0;
        c = new double[0];
        rep = new lsfitreport();
        lsfit.lsfitlinear(y, fmatrix, n, m, ref info, ref c, rep.innerobj);
        return;
    }
    public static void lsfitlinear(double[] y, double[,] fmatrix, out int info, out double[] c, out lsfitreport rep)
    {
        int n;
        int m;
        if( (ap.len(y)!=ap.rows(fmatrix)))
            throw new alglibexception("Error while calling 'lsfitlinear': looks like one of arguments has wrong size");
        info = 0;
        c = new double[0];
        rep = new lsfitreport();
        n = ap.len(y);
        m = ap.cols(fmatrix);
        lsfit.lsfitlinear(y, fmatrix, n, m, ref info, ref c, rep.innerobj);

        return;
    }

    /*************************************************************************
    Constained linear least squares fitting.

    This  is  variation  of LSFitLinear(),  which searchs for min|A*x=b| given
    that  K  additional  constaints  C*x=bc are satisfied. It reduces original
    task to modified one: min|B*y-d| WITHOUT constraints,  then  LSFitLinear()
    is called.

    INPUT PARAMETERS:
        Y       -   array[0..N-1] Function values in  N  points.
        FMatrix -   a table of basis functions values, array[0..N-1, 0..M-1].
                    FMatrix[I,J] - value of J-th basis function in I-th point.
        CMatrix -   a table of constaints, array[0..K-1,0..M].
                    I-th row of CMatrix corresponds to I-th linear constraint:
                    CMatrix[I,0]*C[0] + ... + CMatrix[I,M-1]*C[M-1] = CMatrix[I,M]
        N       -   number of points used. N>=1.
        M       -   number of basis functions, M>=1.
        K       -   number of constraints, 0 <= K < M
                    K=0 corresponds to absence of constraints.

    OUTPUT PARAMETERS:
        Info    -   error code:
                    * -4    internal SVD decomposition subroutine failed (very
                            rare and for degenerate systems only)
                    * -3    either   too   many  constraints  (M   or   more),
                            degenerate  constraints   (some   constraints  are
                            repetead twice) or inconsistent  constraints  were
                            specified.
                    *  1    task is solved
        C       -   decomposition coefficients, array[0..M-1]
        Rep     -   fitting report. Following fields are set:
                    * RMSError          rms error on the (X,Y).
                    * AvgError          average error on the (X,Y).
                    * AvgRelError       average relative error on the non-zero Y
                    * MaxError          maximum error
                                        NON-WEIGHTED ERRORS ARE CALCULATED

    IMPORTANT:
        this subroitine doesn't calculate task's condition number for K<>0.

      -- ALGLIB --
         Copyright 07.09.2009 by Bochkanov Sergey
    *************************************************************************/
    public static void lsfitlinearc(double[] y, double[,] fmatrix, double[,] cmatrix, int n, int m, int k, out int info, out double[] c, out lsfitreport rep)
    {
        info = 0;
        c = new double[0];
        rep = new lsfitreport();
        lsfit.lsfitlinearc(y, fmatrix, cmatrix, n, m, k, ref info, ref c, rep.innerobj);
        return;
    }
    public static void lsfitlinearc(double[] y, double[,] fmatrix, double[,] cmatrix, out int info, out double[] c, out lsfitreport rep)
    {
        int n;
        int m;
        int k;
        if( (ap.len(y)!=ap.rows(fmatrix)))
            throw new alglibexception("Error while calling 'lsfitlinearc': looks like one of arguments has wrong size");
        if( (ap.cols(fmatrix)!=ap.cols(cmatrix)-1))
            throw new alglibexception("Error while calling 'lsfitlinearc': looks like one of arguments has wrong size");
        info = 0;
        c = new double[0];
        rep = new lsfitreport();
        n = ap.len(y);
        m = ap.cols(fmatrix);
        k = ap.rows(cmatrix);
        lsfit.lsfitlinearc(y, fmatrix, cmatrix, n, m, k, ref info, ref c, rep.innerobj);

        return;
    }

    /*************************************************************************
    Weighted nonlinear least squares fitting using function values only.

    Combination of numerical differentiation and secant updates is used to
    obtain function Jacobian.

    Nonlinear task min(F(c)) is solved, where

        F(c) = (w[0]*(f(c,x[0])-y[0]))^2 + ... + (w[n-1]*(f(c,x[n-1])-y[n-1]))^2,

        * N is a number of points,
        * M is a dimension of a space points belong to,
        * K is a dimension of a space of parameters being fitted,
        * w is an N-dimensional vector of weight coefficients,
        * x is a set of N points, each of them is an M-dimensional vector,
        * c is a K-dimensional vector of parameters being fitted

    This subroutine uses only f(c,x[i]).

    INPUT PARAMETERS:
        X       -   array[0..N-1,0..M-1], points (one row = one point)
        Y       -   array[0..N-1], function values.
        W       -   weights, array[0..N-1]
        C       -   array[0..K-1], initial approximation to the solution,
        N       -   number of points, N>1
        M       -   dimension of space
        K       -   number of parameters being fitted
        DiffStep-   numerical differentiation step;
                    should not be very small or large;
                    large = loss of accuracy
                    small = growth of round-off errors

    OUTPUT PARAMETERS:
        State   -   structure which stores algorithm state

      -- ALGLIB --
         Copyright 18.10.2008 by Bochkanov Sergey
    *************************************************************************/
    public static void lsfitcreatewf(double[,] x, double[] y, double[] w, double[] c, int n, int m, int k, double diffstep, out lsfitstate state)
    {
        state = new lsfitstate();
        lsfit.lsfitcreatewf(x, y, w, c, n, m, k, diffstep, state.innerobj);
        return;
    }
    public static void lsfitcreatewf(double[,] x, double[] y, double[] w, double[] c, double diffstep, out lsfitstate state)
    {
        int n;
        int m;
        int k;
        if( (ap.rows(x)!=ap.len(y)) || (ap.rows(x)!=ap.len(w)))
            throw new alglibexception("Error while calling 'lsfitcreatewf': looks like one of arguments has wrong size");
        state = new lsfitstate();
        n = ap.rows(x);
        m = ap.cols(x);
        k = ap.len(c);
        lsfit.lsfitcreatewf(x, y, w, c, n, m, k, diffstep, state.innerobj);

        return;
    }

    /*************************************************************************
    Nonlinear least squares fitting using function values only.

    Combination of numerical differentiation and secant updates is used to
    obtain function Jacobian.

    Nonlinear task min(F(c)) is solved, where

        F(c) = (f(c,x[0])-y[0])^2 + ... + (f(c,x[n-1])-y[n-1])^2,

        * N is a number of points,
        * M is a dimension of a space points belong to,
        * K is a dimension of a space of parameters being fitted,
        * w is an N-dimensional vector of weight coefficients,
        * x is a set of N points, each of them is an M-dimensional vector,
        * c is a K-dimensional vector of parameters being fitted

    This subroutine uses only f(c,x[i]).

    INPUT PARAMETERS:
        X       -   array[0..N-1,0..M-1], points (one row = one point)
        Y       -   array[0..N-1], function values.
        C       -   array[0..K-1], initial approximation to the solution,
        N       -   number of points, N>1
        M       -   dimension of space
        K       -   number of parameters being fitted
        DiffStep-   numerical differentiation step;
                    should not be very small or large;
                    large = loss of accuracy
                    small = growth of round-off errors

    OUTPUT PARAMETERS:
        State   -   structure which stores algorithm state

      -- ALGLIB --
         Copyright 18.10.2008 by Bochkanov Sergey
    *************************************************************************/
    public static void lsfitcreatef(double[,] x, double[] y, double[] c, int n, int m, int k, double diffstep, out lsfitstate state)
    {
        state = new lsfitstate();
        lsfit.lsfitcreatef(x, y, c, n, m, k, diffstep, state.innerobj);
        return;
    }
    public static void lsfitcreatef(double[,] x, double[] y, double[] c, double diffstep, out lsfitstate state)
    {
        int n;
        int m;
        int k;
        if( (ap.rows(x)!=ap.len(y)))
            throw new alglibexception("Error while calling 'lsfitcreatef': looks like one of arguments has wrong size");
        state = new lsfitstate();
        n = ap.rows(x);
        m = ap.cols(x);
        k = ap.len(c);
        lsfit.lsfitcreatef(x, y, c, n, m, k, diffstep, state.innerobj);

        return;
    }

    /*************************************************************************
    Weighted nonlinear least squares fitting using gradient only.

    Nonlinear task min(F(c)) is solved, where

        F(c) = (w[0]*(f(c,x[0])-y[0]))^2 + ... + (w[n-1]*(f(c,x[n-1])-y[n-1]))^2,

        * N is a number of points,
        * M is a dimension of a space points belong to,
        * K is a dimension of a space of parameters being fitted,
        * w is an N-dimensional vector of weight coefficients,
        * x is a set of N points, each of them is an M-dimensional vector,
        * c is a K-dimensional vector of parameters being fitted

    This subroutine uses only f(c,x[i]) and its gradient.

    INPUT PARAMETERS:
        X       -   array[0..N-1,0..M-1], points (one row = one point)
        Y       -   array[0..N-1], function values.
        W       -   weights, array[0..N-1]
        C       -   array[0..K-1], initial approximation to the solution,
        N       -   number of points, N>1
        M       -   dimension of space
        K       -   number of parameters being fitted
        CheapFG -   boolean flag, which is:
                    * True  if both function and gradient calculation complexity
                            are less than O(M^2).  An improved  algorithm  can
                            be  used  which corresponds  to  FGJ  scheme  from
                            MINLM unit.
                    * False otherwise.
                            Standard Jacibian-bases  Levenberg-Marquardt  algo
                            will be used (FJ scheme).

    OUTPUT PARAMETERS:
        State   -   structure which stores algorithm state

    See also:
        LSFitResults
        LSFitCreateFG (fitting without weights)
        LSFitCreateWFGH (fitting using Hessian)
        LSFitCreateFGH (fitting using Hessian, without weights)

      -- ALGLIB --
         Copyright 17.08.2009 by Bochkanov Sergey
    *************************************************************************/
    public static void lsfitcreatewfg(double[,] x, double[] y, double[] w, double[] c, int n, int m, int k, bool cheapfg, out lsfitstate state)
    {
        state = new lsfitstate();
        lsfit.lsfitcreatewfg(x, y, w, c, n, m, k, cheapfg, state.innerobj);
        return;
    }
    public static void lsfitcreatewfg(double[,] x, double[] y, double[] w, double[] c, bool cheapfg, out lsfitstate state)
    {
        int n;
        int m;
        int k;
        if( (ap.rows(x)!=ap.len(y)) || (ap.rows(x)!=ap.len(w)))
            throw new alglibexception("Error while calling 'lsfitcreatewfg': looks like one of arguments has wrong size");
        state = new lsfitstate();
        n = ap.rows(x);
        m = ap.cols(x);
        k = ap.len(c);
        lsfit.lsfitcreatewfg(x, y, w, c, n, m, k, cheapfg, state.innerobj);

        return;
    }

    /*************************************************************************
    Nonlinear least squares fitting using gradient only, without individual
    weights.

    Nonlinear task min(F(c)) is solved, where

        F(c) = ((f(c,x[0])-y[0]))^2 + ... + ((f(c,x[n-1])-y[n-1]))^2,

        * N is a number of points,
        * M is a dimension of a space points belong to,
        * K is a dimension of a space of parameters being fitted,
        * x is a set of N points, each of them is an M-dimensional vector,
        * c is a K-dimensional vector of parameters being fitted

    This subroutine uses only f(c,x[i]) and its gradient.

    INPUT PARAMETERS:
        X       -   array[0..N-1,0..M-1], points (one row = one point)
        Y       -   array[0..N-1], function values.
        C       -   array[0..K-1], initial approximation to the solution,
        N       -   number of points, N>1
        M       -   dimension of space
        K       -   number of parameters being fitted
        CheapFG -   boolean flag, which is:
                    * True  if both function and gradient calculation complexity
                            are less than O(M^2).  An improved  algorithm  can
                            be  used  which corresponds  to  FGJ  scheme  from
                            MINLM unit.
                    * False otherwise.
                            Standard Jacibian-bases  Levenberg-Marquardt  algo
                            will be used (FJ scheme).

    OUTPUT PARAMETERS:
        State   -   structure which stores algorithm state

      -- ALGLIB --
         Copyright 17.08.2009 by Bochkanov Sergey
    *************************************************************************/
    public static void lsfitcreatefg(double[,] x, double[] y, double[] c, int n, int m, int k, bool cheapfg, out lsfitstate state)
    {
        state = new lsfitstate();
        lsfit.lsfitcreatefg(x, y, c, n, m, k, cheapfg, state.innerobj);
        return;
    }
    public static void lsfitcreatefg(double[,] x, double[] y, double[] c, bool cheapfg, out lsfitstate state)
    {
        int n;
        int m;
        int k;
        if( (ap.rows(x)!=ap.len(y)))
            throw new alglibexception("Error while calling 'lsfitcreatefg': looks like one of arguments has wrong size");
        state = new lsfitstate();
        n = ap.rows(x);
        m = ap.cols(x);
        k = ap.len(c);
        lsfit.lsfitcreatefg(x, y, c, n, m, k, cheapfg, state.innerobj);

        return;
    }

    /*************************************************************************
    Weighted nonlinear least squares fitting using gradient/Hessian.

    Nonlinear task min(F(c)) is solved, where

        F(c) = (w[0]*(f(c,x[0])-y[0]))^2 + ... + (w[n-1]*(f(c,x[n-1])-y[n-1]))^2,

        * N is a number of points,
        * M is a dimension of a space points belong to,
        * K is a dimension of a space of parameters being fitted,
        * w is an N-dimensional vector of weight coefficients,
        * x is a set of N points, each of them is an M-dimensional vector,
        * c is a K-dimensional vector of parameters being fitted

    This subroutine uses f(c,x[i]), its gradient and its Hessian.

    INPUT PARAMETERS:
        X       -   array[0..N-1,0..M-1], points (one row = one point)
        Y       -   array[0..N-1], function values.
        W       -   weights, array[0..N-1]
        C       -   array[0..K-1], initial approximation to the solution,
        N       -   number of points, N>1
        M       -   dimension of space
        K       -   number of parameters being fitted

    OUTPUT PARAMETERS:
        State   -   structure which stores algorithm state

      -- ALGLIB --
         Copyright 17.08.2009 by Bochkanov Sergey
    *************************************************************************/
    public static void lsfitcreatewfgh(double[,] x, double[] y, double[] w, double[] c, int n, int m, int k, out lsfitstate state)
    {
        state = new lsfitstate();
        lsfit.lsfitcreatewfgh(x, y, w, c, n, m, k, state.innerobj);
        return;
    }
    public static void lsfitcreatewfgh(double[,] x, double[] y, double[] w, double[] c, out lsfitstate state)
    {
        int n;
        int m;
        int k;
        if( (ap.rows(x)!=ap.len(y)) || (ap.rows(x)!=ap.len(w)))
            throw new alglibexception("Error while calling 'lsfitcreatewfgh': looks like one of arguments has wrong size");
        state = new lsfitstate();
        n = ap.rows(x);
        m = ap.cols(x);
        k = ap.len(c);
        lsfit.lsfitcreatewfgh(x, y, w, c, n, m, k, state.innerobj);

        return;
    }

    /*************************************************************************
    Nonlinear least squares fitting using gradient/Hessian, without individial
    weights.

    Nonlinear task min(F(c)) is solved, where

        F(c) = ((f(c,x[0])-y[0]))^2 + ... + ((f(c,x[n-1])-y[n-1]))^2,

        * N is a number of points,
        * M is a dimension of a space points belong to,
        * K is a dimension of a space of parameters being fitted,
        * x is a set of N points, each of them is an M-dimensional vector,
        * c is a K-dimensional vector of parameters being fitted

    This subroutine uses f(c,x[i]), its gradient and its Hessian.

    INPUT PARAMETERS:
        X       -   array[0..N-1,0..M-1], points (one row = one point)
        Y       -   array[0..N-1], function values.
        C       -   array[0..K-1], initial approximation to the solution,
        N       -   number of points, N>1
        M       -   dimension of space
        K       -   number of parameters being fitted

    OUTPUT PARAMETERS:
        State   -   structure which stores algorithm state


      -- ALGLIB --
         Copyright 17.08.2009 by Bochkanov Sergey
    *************************************************************************/
    public static void lsfitcreatefgh(double[,] x, double[] y, double[] c, int n, int m, int k, out lsfitstate state)
    {
        state = new lsfitstate();
        lsfit.lsfitcreatefgh(x, y, c, n, m, k, state.innerobj);
        return;
    }
    public static void lsfitcreatefgh(double[,] x, double[] y, double[] c, out lsfitstate state)
    {
        int n;
        int m;
        int k;
        if( (ap.rows(x)!=ap.len(y)))
            throw new alglibexception("Error while calling 'lsfitcreatefgh': looks like one of arguments has wrong size");
        state = new lsfitstate();
        n = ap.rows(x);
        m = ap.cols(x);
        k = ap.len(c);
        lsfit.lsfitcreatefgh(x, y, c, n, m, k, state.innerobj);

        return;
    }

    /*************************************************************************
    Stopping conditions for nonlinear least squares fitting.

    INPUT PARAMETERS:
        State   -   structure which stores algorithm state
        EpsF    -   stopping criterion. Algorithm stops if
                    |F(k+1)-F(k)| <= EpsF*max{|F(k)|, |F(k+1)|, 1}
        EpsX    -   >=0
                    The subroutine finishes its work if  on  k+1-th  iteration
                    the condition |v|<=EpsX is fulfilled, where:
                    * |.| means Euclidian norm
                    * v - scaled step vector, v[i]=dx[i]/s[i]
                    * dx - ste pvector, dx=X(k+1)-X(k)
                    * s - scaling coefficients set by LSFitSetScale()
        MaxIts  -   maximum number of iterations. If MaxIts=0, the  number  of
                    iterations   is    unlimited.   Only   Levenberg-Marquardt
                    iterations  are  counted  (L-BFGS/CG  iterations  are  NOT
                    counted because their cost is very low compared to that of
                    LM).

    NOTE

    Passing EpsF=0, EpsX=0 and MaxIts=0 (simultaneously) will lead to automatic
    stopping criterion selection (according to the scheme used by MINLM unit).


      -- ALGLIB --
         Copyright 17.08.2009 by Bochkanov Sergey
    *************************************************************************/
    public static void lsfitsetcond(lsfitstate state, double epsf, double epsx, int maxits)
    {

        lsfit.lsfitsetcond(state.innerobj, epsf, epsx, maxits);
        return;
    }

    /*************************************************************************
    This function sets maximum step length

    INPUT PARAMETERS:
        State   -   structure which stores algorithm state
        StpMax  -   maximum step length, >=0. Set StpMax to 0.0,  if you don't
                    want to limit step length.

    Use this subroutine when you optimize target function which contains exp()
    or  other  fast  growing  functions,  and optimization algorithm makes too
    large  steps  which  leads  to overflow. This function allows us to reject
    steps  that  are  too  large  (and  therefore  expose  us  to the possible
    overflow) without actually calculating function value at the x+stp*d.

    NOTE: non-zero StpMax leads to moderate  performance  degradation  because
    intermediate  step  of  preconditioned L-BFGS optimization is incompatible
    with limits on step size.

      -- ALGLIB --
         Copyright 02.04.2010 by Bochkanov Sergey
    *************************************************************************/
    public static void lsfitsetstpmax(lsfitstate state, double stpmax)
    {

        lsfit.lsfitsetstpmax(state.innerobj, stpmax);
        return;
    }

    /*************************************************************************
    This function turns on/off reporting.

    INPUT PARAMETERS:
        State   -   structure which stores algorithm state
        NeedXRep-   whether iteration reports are needed or not

    When reports are needed, State.C (current parameters) and State.F (current
    value of fitting function) are reported.


      -- ALGLIB --
         Copyright 15.08.2010 by Bochkanov Sergey
    *************************************************************************/
    public static void lsfitsetxrep(lsfitstate state, bool needxrep)
    {

        lsfit.lsfitsetxrep(state.innerobj, needxrep);
        return;
    }

    /*************************************************************************
    This function sets scaling coefficients for underlying optimizer.

    ALGLIB optimizers use scaling matrices to test stopping  conditions  (step
    size and gradient are scaled before comparison with tolerances).  Scale of
    the I-th variable is a translation invariant measure of:
    a) "how large" the variable is
    b) how large the step should be to make significant changes in the function

    Generally, scale is NOT considered to be a form of preconditioner.  But LM
    optimizer is unique in that it uses scaling matrix both  in  the  stopping
    condition tests and as Marquardt damping factor.

    Proper scaling is very important for the algorithm performance. It is less
    important for the quality of results, but still has some influence (it  is
    easier  to  converge  when  variables  are  properly  scaled, so premature
    stopping is possible when very badly scalled variables are  combined  with
    relaxed stopping conditions).

    INPUT PARAMETERS:
        State   -   structure stores algorithm state
        S       -   array[N], non-zero scaling coefficients
                    S[i] may be negative, sign doesn't matter.

      -- ALGLIB --
         Copyright 14.01.2011 by Bochkanov Sergey
    *************************************************************************/
    public static void lsfitsetscale(lsfitstate state, double[] s)
    {

        lsfit.lsfitsetscale(state.innerobj, s);
        return;
    }

    /*************************************************************************
    This function sets boundary constraints for underlying optimizer

    Boundary constraints are inactive by default (after initial creation).
    They are preserved until explicitly turned off with another SetBC() call.

    INPUT PARAMETERS:
        State   -   structure stores algorithm state
        BndL    -   lower bounds, array[K].
                    If some (all) variables are unbounded, you may specify
                    very small number or -INF (latter is recommended because
                    it will allow solver to use better algorithm).
        BndU    -   upper bounds, array[K].
                    If some (all) variables are unbounded, you may specify
                    very large number or +INF (latter is recommended because
                    it will allow solver to use better algorithm).

    NOTE 1: it is possible to specify BndL[i]=BndU[i]. In this case I-th
    variable will be "frozen" at X[i]=BndL[i]=BndU[i].

    NOTE 2: unlike other constrained optimization algorithms, this solver  has
    following useful properties:
    * bound constraints are always satisfied exactly
    * function is evaluated only INSIDE area specified by bound constraints

      -- ALGLIB --
         Copyright 14.01.2011 by Bochkanov Sergey
    *************************************************************************/
    public static void lsfitsetbc(lsfitstate state, double[] bndl, double[] bndu)
    {

        lsfit.lsfitsetbc(state.innerobj, bndl, bndu);
        return;
    }

    /*************************************************************************
    This function provides reverse communication interface
    Reverse communication interface is not documented or recommended to use.
    See below for functions which provide better documented API
    *************************************************************************/
    public static bool lsfititeration(lsfitstate state)
    {

        bool result = lsfit.lsfititeration(state.innerobj);
        return result;
    }
    /*************************************************************************
    This family of functions is used to launcn iterations of nonlinear fitter

    These functions accept following parameters:
        func    -   callback which calculates function (or merit function)
                    value func at given point x
        grad    -   callback which calculates function (or merit function)
                    value func and gradient grad at given point x
        hess    -   callback which calculates function (or merit function)
                    value func, gradient grad and Hessian hess at given point x
        rep     -   optional callback which is called after each iteration
                    can be null
        obj     -   optional object which is passed to func/grad/hess/jac/rep
                    can be null

    NOTES:

    1. this algorithm is somewhat unusual because it works with  parameterized
       function f(C,X), where X is a function argument (we  have  many  points
       which are characterized by different  argument  values),  and  C  is  a
       parameter to fit.

       For example, if we want to do linear fit by f(c0,c1,x) = c0*x+c1,  then
       x will be argument, and {c0,c1} will be parameters.

       It is important to understand that this algorithm finds minimum in  the
       space of function PARAMETERS (not arguments), so it  needs  derivatives
       of f() with respect to C, not X.

       In the example above it will need f=c0*x+c1 and {df/dc0,df/dc1} = {x,1}
       instead of {df/dx} = {c0}.

    2. Callback functions accept C as the first parameter, and X as the second

    3. If  state  was  created  with  LSFitCreateFG(),  algorithm  needs  just
       function   and   its   gradient,   but   if   state   was  created with
       LSFitCreateFGH(), algorithm will need function, gradient and Hessian.

       According  to  the  said  above,  there  ase  several  versions of this
       function, which accept different sets of callbacks.

       This flexibility opens way to subtle errors - you may create state with
       LSFitCreateFGH() (optimization using Hessian), but call function  which
       does not accept Hessian. So when algorithm will request Hessian,  there
       will be no callback to call. In this case exception will be thrown.

       Be careful to avoid such errors because there is no way to find them at
       compile time - you can see them at runtime only.

      -- ALGLIB --
         Copyright 17.08.2009 by Bochkanov Sergey

    *************************************************************************/
    public static void lsfitfit(lsfitstate state, ndimensional_pfunc func, ndimensional_rep rep, object obj)
    {
        if( func==null )
            throw new alglibexception("ALGLIB: error in 'lsfitfit()' (func is null)");
        while( alglib.lsfititeration(state) )
        {
            if( state.needf )
            {
                func(state.c, state.x, ref state.innerobj.f, obj);
                continue;
            }
            if( state.innerobj.xupdated )
            {
                if( rep!=null )
                    rep(state.innerobj.c, state.innerobj.f, obj);
                continue;
            }
            throw new alglibexception("ALGLIB: error in 'lsfitfit' (some derivatives were not provided?)");
        }
    }


    public static void lsfitfit(lsfitstate state, ndimensional_pfunc func, ndimensional_pgrad grad, ndimensional_rep rep, object obj)
    {
        if( func==null )
            throw new alglibexception("ALGLIB: error in 'lsfitfit()' (func is null)");
        if( grad==null )
            throw new alglibexception("ALGLIB: error in 'lsfitfit()' (grad is null)");
        while( alglib.lsfititeration(state) )
        {
            if( state.needf )
            {
                func(state.c, state.x, ref state.innerobj.f, obj);
                continue;
            }
            if( state.needfg )
            {
                grad(state.c, state.x, ref state.innerobj.f, state.innerobj.g, obj);
                continue;
            }
            if( state.innerobj.xupdated )
            {
                if( rep!=null )
                    rep(state.innerobj.c, state.innerobj.f, obj);
                continue;
            }
            throw new alglibexception("ALGLIB: error in 'lsfitfit' (some derivatives were not provided?)");
        }
    }


    public static void lsfitfit(lsfitstate state, ndimensional_pfunc func, ndimensional_pgrad grad, ndimensional_phess hess, ndimensional_rep rep, object obj)
    {
        if( func==null )
            throw new alglibexception("ALGLIB: error in 'lsfitfit()' (func is null)");
        if( grad==null )
            throw new alglibexception("ALGLIB: error in 'lsfitfit()' (grad is null)");
        if( hess==null )
            throw new alglibexception("ALGLIB: error in 'lsfitfit()' (hess is null)");
        while( alglib.lsfititeration(state) )
        {
            if( state.needf )
            {
                func(state.c, state.x, ref state.innerobj.f, obj);
                continue;
            }
            if( state.needfg )
            {
                grad(state.c, state.x, ref state.innerobj.f, state.innerobj.g, obj);
                continue;
            }
            if( state.needfgh )
            {
                hess(state.c, state.x, ref state.innerobj.f, state.innerobj.g, state.innerobj.h, obj);
                continue;
            }
            if( state.innerobj.xupdated )
            {
                if( rep!=null )
                    rep(state.innerobj.c, state.innerobj.f, obj);
                continue;
            }
            throw new alglibexception("ALGLIB: error in 'lsfitfit' (some derivatives were not provided?)");
        }
    }



    /*************************************************************************
    Nonlinear least squares fitting results.

    Called after return from LSFitFit().

    INPUT PARAMETERS:
        State   -   algorithm state

    OUTPUT PARAMETERS:
        Info    -   completetion code:
                        *  1    relative function improvement is no more than
                                EpsF.
                        *  2    relative step is no more than EpsX.
                        *  4    gradient norm is no more than EpsG
                        *  5    MaxIts steps was taken
                        *  7    stopping conditions are too stringent,
                                further improvement is impossible
        C       -   array[0..K-1], solution
        Rep     -   optimization report. Following fields are set:
                    * Rep.TerminationType completetion code:
                    * RMSError          rms error on the (X,Y).
                    * AvgError          average error on the (X,Y).
                    * AvgRelError       average relative error on the non-zero Y
                    * MaxError          maximum error
                                        NON-WEIGHTED ERRORS ARE CALCULATED
                    * WRMSError         weighted rms error on the (X,Y).


      -- ALGLIB --
         Copyright 17.08.2009 by Bochkanov Sergey
    *************************************************************************/
    public static void lsfitresults(lsfitstate state, out int info, out double[] c, out lsfitreport rep)
    {
        info = 0;
        c = new double[0];
        rep = new lsfitreport();
        lsfit.lsfitresults(state.innerobj, ref info, ref c, rep.innerobj);
        return;
    }

}
public partial class alglib
{


    /*************************************************************************
    Parametric spline inteprolant: 2-dimensional curve.

    You should not try to access its members directly - use PSpline2XXXXXXXX()
    functions instead.
    *************************************************************************/
    public class pspline2interpolant
    {
        //
        // Public declarations
        //

        public pspline2interpolant()
        {
            _innerobj = new pspline.pspline2interpolant();
        }

        //
        // Although some of declarations below are public, you should not use them
        // They are intended for internal use only
        //
        private pspline.pspline2interpolant _innerobj;
        public pspline.pspline2interpolant innerobj { get { return _innerobj; } }
        public pspline2interpolant(pspline.pspline2interpolant obj)
        {
            _innerobj = obj;
        }
    }


    /*************************************************************************
    Parametric spline inteprolant: 3-dimensional curve.

    You should not try to access its members directly - use PSpline3XXXXXXXX()
    functions instead.
    *************************************************************************/
    public class pspline3interpolant
    {
        //
        // Public declarations
        //

        public pspline3interpolant()
        {
            _innerobj = new pspline.pspline3interpolant();
        }

        //
        // Although some of declarations below are public, you should not use them
        // They are intended for internal use only
        //
        private pspline.pspline3interpolant _innerobj;
        public pspline.pspline3interpolant innerobj { get { return _innerobj; } }
        public pspline3interpolant(pspline.pspline3interpolant obj)
        {
            _innerobj = obj;
        }
    }

    /*************************************************************************
    This function  builds  non-periodic 2-dimensional parametric spline  which
    starts at (X[0],Y[0]) and ends at (X[N-1],Y[N-1]).

    INPUT PARAMETERS:
        XY  -   points, array[0..N-1,0..1].
                XY[I,0:1] corresponds to the Ith point.
                Order of points is important!
        N   -   points count, N>=5 for Akima splines, N>=2 for other types  of
                splines.
        ST  -   spline type:
                * 0     Akima spline
                * 1     parabolically terminated Catmull-Rom spline (Tension=0)
                * 2     parabolically terminated cubic spline
        PT  -   parameterization type:
                * 0     uniform
                * 1     chord length
                * 2     centripetal

    OUTPUT PARAMETERS:
        P   -   parametric spline interpolant


    NOTES:
    * this function  assumes  that  there all consequent points  are distinct.
      I.e. (x0,y0)<>(x1,y1),  (x1,y1)<>(x2,y2),  (x2,y2)<>(x3,y3)  and  so on.
      However, non-consequent points may coincide, i.e. we can  have  (x0,y0)=
      =(x2,y2).

      -- ALGLIB PROJECT --
         Copyright 28.05.2010 by Bochkanov Sergey
    *************************************************************************/
    public static void pspline2build(double[,] xy, int n, int st, int pt, out pspline2interpolant p)
    {
        p = new pspline2interpolant();
        pspline.pspline2build(xy, n, st, pt, p.innerobj);
        return;
    }

    /*************************************************************************
    This function  builds  non-periodic 3-dimensional parametric spline  which
    starts at (X[0],Y[0],Z[0]) and ends at (X[N-1],Y[N-1],Z[N-1]).

    Same as PSpline2Build() function, but for 3D, so we  won't  duplicate  its
    description here.

      -- ALGLIB PROJECT --
         Copyright 28.05.2010 by Bochkanov Sergey
    *************************************************************************/
    public static void pspline3build(double[,] xy, int n, int st, int pt, out pspline3interpolant p)
    {
        p = new pspline3interpolant();
        pspline.pspline3build(xy, n, st, pt, p.innerobj);
        return;
    }

    /*************************************************************************
    This  function  builds  periodic  2-dimensional  parametric  spline  which
    starts at (X[0],Y[0]), goes through all points to (X[N-1],Y[N-1]) and then
    back to (X[0],Y[0]).

    INPUT PARAMETERS:
        XY  -   points, array[0..N-1,0..1].
                XY[I,0:1] corresponds to the Ith point.
                XY[N-1,0:1] must be different from XY[0,0:1].
                Order of points is important!
        N   -   points count, N>=3 for other types of splines.
        ST  -   spline type:
                * 1     Catmull-Rom spline (Tension=0) with cyclic boundary conditions
                * 2     cubic spline with cyclic boundary conditions
        PT  -   parameterization type:
                * 0     uniform
                * 1     chord length
                * 2     centripetal

    OUTPUT PARAMETERS:
        P   -   parametric spline interpolant


    NOTES:
    * this function  assumes  that there all consequent points  are  distinct.
      I.e. (x0,y0)<>(x1,y1), (x1,y1)<>(x2,y2),  (x2,y2)<>(x3,y3)  and  so  on.
      However, non-consequent points may coincide, i.e. we can  have  (x0,y0)=
      =(x2,y2).
    * last point of sequence is NOT equal to the first  point.  You  shouldn't
      make curve "explicitly periodic" by making them equal.

      -- ALGLIB PROJECT --
         Copyright 28.05.2010 by Bochkanov Sergey
    *************************************************************************/
    public static void pspline2buildperiodic(double[,] xy, int n, int st, int pt, out pspline2interpolant p)
    {
        p = new pspline2interpolant();
        pspline.pspline2buildperiodic(xy, n, st, pt, p.innerobj);
        return;
    }

    /*************************************************************************
    This  function  builds  periodic  3-dimensional  parametric  spline  which
    starts at (X[0],Y[0],Z[0]), goes through all points to (X[N-1],Y[N-1],Z[N-1])
    and then back to (X[0],Y[0],Z[0]).

    Same as PSpline2Build() function, but for 3D, so we  won't  duplicate  its
    description here.

      -- ALGLIB PROJECT --
         Copyright 28.05.2010 by Bochkanov Sergey
    *************************************************************************/
    public static void pspline3buildperiodic(double[,] xy, int n, int st, int pt, out pspline3interpolant p)
    {
        p = new pspline3interpolant();
        pspline.pspline3buildperiodic(xy, n, st, pt, p.innerobj);
        return;
    }

    /*************************************************************************
    This function returns vector of parameter values correspoding to points.

    I.e. for P created from (X[0],Y[0])...(X[N-1],Y[N-1]) and U=TValues(P)  we
    have
        (X[0],Y[0]) = PSpline2Calc(P,U[0]),
        (X[1],Y[1]) = PSpline2Calc(P,U[1]),
        (X[2],Y[2]) = PSpline2Calc(P,U[2]),
        ...

    INPUT PARAMETERS:
        P   -   parametric spline interpolant

    OUTPUT PARAMETERS:
        N   -   array size
        T   -   array[0..N-1]


    NOTES:
    * for non-periodic splines U[0]=0, U[0]<U[1]<...<U[N-1], U[N-1]=1
    * for periodic splines     U[0]=0, U[0]<U[1]<...<U[N-1], U[N-1]<1

      -- ALGLIB PROJECT --
         Copyright 28.05.2010 by Bochkanov Sergey
    *************************************************************************/
    public static void pspline2parametervalues(pspline2interpolant p, out int n, out double[] t)
    {
        n = 0;
        t = new double[0];
        pspline.pspline2parametervalues(p.innerobj, ref n, ref t);
        return;
    }

    /*************************************************************************
    This function returns vector of parameter values correspoding to points.

    Same as PSpline2ParameterValues(), but for 3D.

      -- ALGLIB PROJECT --
         Copyright 28.05.2010 by Bochkanov Sergey
    *************************************************************************/
    public static void pspline3parametervalues(pspline3interpolant p, out int n, out double[] t)
    {
        n = 0;
        t = new double[0];
        pspline.pspline3parametervalues(p.innerobj, ref n, ref t);
        return;
    }

    /*************************************************************************
    This function  calculates  the value of the parametric spline for a  given
    value of parameter T

    INPUT PARAMETERS:
        P   -   parametric spline interpolant
        T   -   point:
                * T in [0,1] corresponds to interval spanned by points
                * for non-periodic splines T<0 (or T>1) correspond to parts of
                  the curve before the first (after the last) point
                * for periodic splines T<0 (or T>1) are projected  into  [0,1]
                  by making T=T-floor(T).

    OUTPUT PARAMETERS:
        X   -   X-position
        Y   -   Y-position


      -- ALGLIB PROJECT --
         Copyright 28.05.2010 by Bochkanov Sergey
    *************************************************************************/
    public static void pspline2calc(pspline2interpolant p, double t, out double x, out double y)
    {
        x = 0;
        y = 0;
        pspline.pspline2calc(p.innerobj, t, ref x, ref y);
        return;
    }

    /*************************************************************************
    This function  calculates  the value of the parametric spline for a  given
    value of parameter T.

    INPUT PARAMETERS:
        P   -   parametric spline interpolant
        T   -   point:
                * T in [0,1] corresponds to interval spanned by points
                * for non-periodic splines T<0 (or T>1) correspond to parts of
                  the curve before the first (after the last) point
                * for periodic splines T<0 (or T>1) are projected  into  [0,1]
                  by making T=T-floor(T).

    OUTPUT PARAMETERS:
        X   -   X-position
        Y   -   Y-position
        Z   -   Z-position


      -- ALGLIB PROJECT --
         Copyright 28.05.2010 by Bochkanov Sergey
    *************************************************************************/
    public static void pspline3calc(pspline3interpolant p, double t, out double x, out double y, out double z)
    {
        x = 0;
        y = 0;
        z = 0;
        pspline.pspline3calc(p.innerobj, t, ref x, ref y, ref z);
        return;
    }

    /*************************************************************************
    This function  calculates  tangent vector for a given value of parameter T

    INPUT PARAMETERS:
        P   -   parametric spline interpolant
        T   -   point:
                * T in [0,1] corresponds to interval spanned by points
                * for non-periodic splines T<0 (or T>1) correspond to parts of
                  the curve before the first (after the last) point
                * for periodic splines T<0 (or T>1) are projected  into  [0,1]
                  by making T=T-floor(T).

    OUTPUT PARAMETERS:
        X    -   X-component of tangent vector (normalized)
        Y    -   Y-component of tangent vector (normalized)

    NOTE:
        X^2+Y^2 is either 1 (for non-zero tangent vector) or 0.


      -- ALGLIB PROJECT --
         Copyright 28.05.2010 by Bochkanov Sergey
    *************************************************************************/
    public static void pspline2tangent(pspline2interpolant p, double t, out double x, out double y)
    {
        x = 0;
        y = 0;
        pspline.pspline2tangent(p.innerobj, t, ref x, ref y);
        return;
    }

    /*************************************************************************
    This function  calculates  tangent vector for a given value of parameter T

    INPUT PARAMETERS:
        P   -   parametric spline interpolant
        T   -   point:
                * T in [0,1] corresponds to interval spanned by points
                * for non-periodic splines T<0 (or T>1) correspond to parts of
                  the curve before the first (after the last) point
                * for periodic splines T<0 (or T>1) are projected  into  [0,1]
                  by making T=T-floor(T).

    OUTPUT PARAMETERS:
        X    -   X-component of tangent vector (normalized)
        Y    -   Y-component of tangent vector (normalized)
        Z    -   Z-component of tangent vector (normalized)

    NOTE:
        X^2+Y^2+Z^2 is either 1 (for non-zero tangent vector) or 0.


      -- ALGLIB PROJECT --
         Copyright 28.05.2010 by Bochkanov Sergey
    *************************************************************************/
    public static void pspline3tangent(pspline3interpolant p, double t, out double x, out double y, out double z)
    {
        x = 0;
        y = 0;
        z = 0;
        pspline.pspline3tangent(p.innerobj, t, ref x, ref y, ref z);
        return;
    }

    /*************************************************************************
    This function calculates derivative, i.e. it returns (dX/dT,dY/dT).

    INPUT PARAMETERS:
        P   -   parametric spline interpolant
        T   -   point:
                * T in [0,1] corresponds to interval spanned by points
                * for non-periodic splines T<0 (or T>1) correspond to parts of
                  the curve before the first (after the last) point
                * for periodic splines T<0 (or T>1) are projected  into  [0,1]
                  by making T=T-floor(T).

    OUTPUT PARAMETERS:
        X   -   X-value
        DX  -   X-derivative
        Y   -   Y-value
        DY  -   Y-derivative


      -- ALGLIB PROJECT --
         Copyright 28.05.2010 by Bochkanov Sergey
    *************************************************************************/
    public static void pspline2diff(pspline2interpolant p, double t, out double x, out double dx, out double y, out double dy)
    {
        x = 0;
        dx = 0;
        y = 0;
        dy = 0;
        pspline.pspline2diff(p.innerobj, t, ref x, ref dx, ref y, ref dy);
        return;
    }

    /*************************************************************************
    This function calculates derivative, i.e. it returns (dX/dT,dY/dT,dZ/dT).

    INPUT PARAMETERS:
        P   -   parametric spline interpolant
        T   -   point:
                * T in [0,1] corresponds to interval spanned by points
                * for non-periodic splines T<0 (or T>1) correspond to parts of
                  the curve before the first (after the last) point
                * for periodic splines T<0 (or T>1) are projected  into  [0,1]
                  by making T=T-floor(T).

    OUTPUT PARAMETERS:
        X   -   X-value
        DX  -   X-derivative
        Y   -   Y-value
        DY  -   Y-derivative
        Z   -   Z-value
        DZ  -   Z-derivative


      -- ALGLIB PROJECT --
         Copyright 28.05.2010 by Bochkanov Sergey
    *************************************************************************/
    public static void pspline3diff(pspline3interpolant p, double t, out double x, out double dx, out double y, out double dy, out double z, out double dz)
    {
        x = 0;
        dx = 0;
        y = 0;
        dy = 0;
        z = 0;
        dz = 0;
        pspline.pspline3diff(p.innerobj, t, ref x, ref dx, ref y, ref dy, ref z, ref dz);
        return;
    }

    /*************************************************************************
    This function calculates first and second derivative with respect to T.

    INPUT PARAMETERS:
        P   -   parametric spline interpolant
        T   -   point:
                * T in [0,1] corresponds to interval spanned by points
                * for non-periodic splines T<0 (or T>1) correspond to parts of
                  the curve before the first (after the last) point
                * for periodic splines T<0 (or T>1) are projected  into  [0,1]
                  by making T=T-floor(T).

    OUTPUT PARAMETERS:
        X   -   X-value
        DX  -   derivative
        D2X -   second derivative
        Y   -   Y-value
        DY  -   derivative
        D2Y -   second derivative


      -- ALGLIB PROJECT --
         Copyright 28.05.2010 by Bochkanov Sergey
    *************************************************************************/
    public static void pspline2diff2(pspline2interpolant p, double t, out double x, out double dx, out double d2x, out double y, out double dy, out double d2y)
    {
        x = 0;
        dx = 0;
        d2x = 0;
        y = 0;
        dy = 0;
        d2y = 0;
        pspline.pspline2diff2(p.innerobj, t, ref x, ref dx, ref d2x, ref y, ref dy, ref d2y);
        return;
    }

    /*************************************************************************
    This function calculates first and second derivative with respect to T.

    INPUT PARAMETERS:
        P   -   parametric spline interpolant
        T   -   point:
                * T in [0,1] corresponds to interval spanned by points
                * for non-periodic splines T<0 (or T>1) correspond to parts of
                  the curve before the first (after the last) point
                * for periodic splines T<0 (or T>1) are projected  into  [0,1]
                  by making T=T-floor(T).

    OUTPUT PARAMETERS:
        X   -   X-value
        DX  -   derivative
        D2X -   second derivative
        Y   -   Y-value
        DY  -   derivative
        D2Y -   second derivative
        Z   -   Z-value
        DZ  -   derivative
        D2Z -   second derivative


      -- ALGLIB PROJECT --
         Copyright 28.05.2010 by Bochkanov Sergey
    *************************************************************************/
    public static void pspline3diff2(pspline3interpolant p, double t, out double x, out double dx, out double d2x, out double y, out double dy, out double d2y, out double z, out double dz, out double d2z)
    {
        x = 0;
        dx = 0;
        d2x = 0;
        y = 0;
        dy = 0;
        d2y = 0;
        z = 0;
        dz = 0;
        d2z = 0;
        pspline.pspline3diff2(p.innerobj, t, ref x, ref dx, ref d2x, ref y, ref dy, ref d2y, ref z, ref dz, ref d2z);
        return;
    }

    /*************************************************************************
    This function  calculates  arc length, i.e. length of  curve  between  t=a
    and t=b.

    INPUT PARAMETERS:
        P   -   parametric spline interpolant
        A,B -   parameter values corresponding to arc ends:
                * B>A will result in positive length returned
                * B<A will result in negative length returned

    RESULT:
        length of arc starting at T=A and ending at T=B.


      -- ALGLIB PROJECT --
         Copyright 30.05.2010 by Bochkanov Sergey
    *************************************************************************/
    public static double pspline2arclength(pspline2interpolant p, double a, double b)
    {

        double result = pspline.pspline2arclength(p.innerobj, a, b);
        return result;
    }

    /*************************************************************************
    This function  calculates  arc length, i.e. length of  curve  between  t=a
    and t=b.

    INPUT PARAMETERS:
        P   -   parametric spline interpolant
        A,B -   parameter values corresponding to arc ends:
                * B>A will result in positive length returned
                * B<A will result in negative length returned

    RESULT:
        length of arc starting at T=A and ending at T=B.


      -- ALGLIB PROJECT --
         Copyright 30.05.2010 by Bochkanov Sergey
    *************************************************************************/
    public static double pspline3arclength(pspline3interpolant p, double a, double b)
    {

        double result = pspline.pspline3arclength(p.innerobj, a, b);
        return result;
    }

}
public partial class alglib
{


    /*************************************************************************
    RBF model.

    Never try to directly work with fields of this object - always use  ALGLIB
    functions to use this object.
    *************************************************************************/
    public class rbfmodel
    {
        //
        // Public declarations
        //

        public rbfmodel()
        {
            _innerobj = new rbf.rbfmodel();
        }

        //
        // Although some of declarations below are public, you should not use them
        // They are intended for internal use only
        //
        private rbf.rbfmodel _innerobj;
        public rbf.rbfmodel innerobj { get { return _innerobj; } }
        public rbfmodel(rbf.rbfmodel obj)
        {
            _innerobj = obj;
        }
    }


    /*************************************************************************
    RBF solution report:
    * TerminationType   -   termination type, positive values - success,
                            non-positive - failure.
    *************************************************************************/
    public class rbfreport
    {
        //
        // Public declarations
        //
        public int arows { get { return _innerobj.arows; } set { _innerobj.arows = value; } }
        public int acols { get { return _innerobj.acols; } set { _innerobj.acols = value; } }
        public int annz { get { return _innerobj.annz; } set { _innerobj.annz = value; } }
        public int iterationscount { get { return _innerobj.iterationscount; } set { _innerobj.iterationscount = value; } }
        public int nmv { get { return _innerobj.nmv; } set { _innerobj.nmv = value; } }
        public int terminationtype { get { return _innerobj.terminationtype; } set { _innerobj.terminationtype = value; } }

        public rbfreport()
        {
            _innerobj = new rbf.rbfreport();
        }

        //
        // Although some of declarations below are public, you should not use them
        // They are intended for internal use only
        //
        private rbf.rbfreport _innerobj;
        public rbf.rbfreport innerobj { get { return _innerobj; } }
        public rbfreport(rbf.rbfreport obj)
        {
            _innerobj = obj;
        }
    }


    /*************************************************************************
    This function serializes data structure to string.

    Important properties of s_out:
    * it contains alphanumeric characters, dots, underscores, minus signs
    * these symbols are grouped into words, which are separated by spaces
      and Windows-style (CR+LF) newlines
    * although  serializer  uses  spaces and CR+LF as separators, you can 
      replace any separator character by arbitrary combination of spaces,
      tabs, Windows or Unix newlines. It allows flexible reformatting  of
      the  string  in  case you want to include it into text or XML file. 
      But you should not insert separators into the middle of the "words"
      nor you should change case of letters.
    * s_out can be freely moved between 32-bit and 64-bit systems, little
      and big endian machines, and so on. You can serialize structure  on
      32-bit machine and unserialize it on 64-bit one (or vice versa), or
      serialize  it  on  SPARC  and  unserialize  on  x86.  You  can also 
      serialize  it  in  C# version of ALGLIB and unserialize in C++ one, 
      and vice versa.
    *************************************************************************/
    public static void rbfserialize(rbfmodel obj, out string s_out)
    {
        alglib.serializer s = new alglib.serializer();
        s.alloc_start();
        rbf.rbfalloc(s, obj.innerobj);
        s.sstart_str();
        rbf.rbfserialize(s, obj.innerobj);
        s.stop();
        s_out = s.get_string();
    }


    /*************************************************************************
    This function unserializes data structure from string.
    *************************************************************************/
    public static void rbfunserialize(string s_in, out rbfmodel obj)
    {
        alglib.serializer s = new alglib.serializer();
        obj = new rbfmodel();
        s.ustart_str(s_in);
        rbf.rbfunserialize(s, obj.innerobj);
        s.stop();
    }

    /*************************************************************************
    This function creates RBF  model  for  a  scalar (NY=1)  or  vector (NY>1)
    function in a NX-dimensional space (NX=2 or NX=3).

    Newly created model is empty. It can be used for interpolation right after
    creation, but it just returns zeros. You have to add points to the  model,
    tune interpolation settings, and then  call  model  construction  function
    RBFBuildModel() which will update model according to your specification.

    USAGE:
    1. User creates model with RBFCreate()
    2. User adds dataset with RBFSetPoints() (points do NOT have to  be  on  a
       regular grid)
    3. (OPTIONAL) User chooses polynomial term by calling:
       * RBFLinTerm() to set linear term
       * RBFConstTerm() to set constant term
       * RBFZeroTerm() to set zero term
       By default, linear term is used.
    4. User chooses specific RBF algorithm to use: either QNN (RBFSetAlgoQNN)
       or ML (RBFSetAlgoMultiLayer).
    5. User calls RBFBuildModel() function which rebuilds model  according  to
       the specification
    6. User may call RBFCalc() to calculate model value at the specified point,
       RBFGridCalc() to  calculate   model  values at the points of the regular
       grid. User may extract model coefficients with RBFUnpack() call.

    INPUT PARAMETERS:
        NX      -   dimension of the space, NX=2 or NX=3
        NY      -   function dimension, NY>=1

    OUTPUT PARAMETERS:
        S       -   RBF model (initially equals to zero)

    NOTE 1: memory requirements. RBF models require amount of memory  which is
            proportional  to  the  number  of data points. Memory is allocated
            during model construction, but most of this memory is freed  after
            model coefficients are calculated.

            Some approximate estimates for N centers with default settings are
            given below:
            * about 250*N*(sizeof(double)+2*sizeof(int)) bytes  of  memory  is
              needed during model construction stage.
            * about 15*N*sizeof(double) bytes is needed after model is built.
            For example, for N=100000 we may need 0.6 GB of memory  to  build
            model, but just about 0.012 GB to store it.

      -- ALGLIB --
         Copyright 13.12.2011 by Bochkanov Sergey
    *************************************************************************/
    public static void rbfcreate(int nx, int ny, out rbfmodel s)
    {
        s = new rbfmodel();
        rbf.rbfcreate(nx, ny, s.innerobj);
        return;
    }

    /*************************************************************************
    This function adds dataset.

    This function overrides results of the previous calls, i.e. multiple calls
    of this function will result in only the last set being added.

    INPUT PARAMETERS:
        S       -   RBF model, initialized by RBFCreate() call.
        XY      -   points, array[N,NX+NY]. One row corresponds to  one  point
                    in the dataset. First NX elements  are  coordinates,  next
                    NY elements are function values. Array may  be larger than
                    specific,  in  this  case  only leading [N,NX+NY] elements
                    will be used.
        N       -   number of points in the dataset

    After you've added dataset and (optionally) tuned algorithm  settings  you
    should call RBFBuildModel() in order to build a model for you.

    NOTE: this   function  has   some   serialization-related  subtleties.  We
          recommend you to study serialization examples from ALGLIB  Reference
          Manual if you want to perform serialization of your models.


      -- ALGLIB --
         Copyright 13.12.2011 by Bochkanov Sergey
    *************************************************************************/
    public static void rbfsetpoints(rbfmodel s, double[,] xy, int n)
    {

        rbf.rbfsetpoints(s.innerobj, xy, n);
        return;
    }
    public static void rbfsetpoints(rbfmodel s, double[,] xy)
    {
        int n;


        n = ap.rows(xy);
        rbf.rbfsetpoints(s.innerobj, xy, n);

        return;
    }

    /*************************************************************************
    This  function  sets  RBF interpolation algorithm. ALGLIB supports several
    RBF algorithms with different properties.

    This algorithm is called RBF-QNN and  it  is  good  for  point  sets  with
    following properties:
    a) all points are distinct
    b) all points are well separated.
    c) points  distribution  is  approximately  uniform.  There is no "contour
       lines", clusters of points, or other small-scale structures.

    Algorithm description:
    1) interpolation centers are allocated to data points
    2) interpolation radii are calculated as distances to the  nearest centers
       times Q coefficient (where Q is a value from [0.75,1.50]).
    3) after  performing (2) radii are transformed in order to avoid situation
       when single outlier has very large radius and  influences  many  points
       across all dataset. Transformation has following form:
           new_r[i] = min(r[i],Z*median(r[]))
       where r[i] is I-th radius, median()  is a median  radius across  entire
       dataset, Z is user-specified value which controls amount  of  deviation
       from median radius.

    When (a) is violated,  we  will  be unable to build RBF model. When (b) or
    (c) are violated, model will be built, but interpolation quality  will  be
    low. See http://www.alglib.net/interpolation/ for more information on this
    subject.

    This algorithm is used by default.

    Additional Q parameter controls smoothness properties of the RBF basis:
    * Q<0.75 will give perfectly conditioned basis,  but  terrible  smoothness
      properties (RBF interpolant will have sharp peaks around function values)
    * Q around 1.0 gives good balance between smoothness and condition number
    * Q>1.5 will lead to badly conditioned systems and slow convergence of the
      underlying linear solver (although smoothness will be very good)
    * Q>2.0 will effectively make optimizer useless because it won't  converge
      within reasonable amount of iterations. It is possible to set such large
      Q, but it is advised not to do so.

    INPUT PARAMETERS:
        S       -   RBF model, initialized by RBFCreate() call
        Q       -   Q parameter, Q>0, recommended value - 1.0
        Z       -   Z parameter, Z>0, recommended value - 5.0

    NOTE: this   function  has   some   serialization-related  subtleties.  We
          recommend you to study serialization examples from ALGLIB  Reference
          Manual if you want to perform serialization of your models.


      -- ALGLIB --
         Copyright 13.12.2011 by Bochkanov Sergey
    *************************************************************************/
    public static void rbfsetalgoqnn(rbfmodel s, double q, double z)
    {

        rbf.rbfsetalgoqnn(s.innerobj, q, z);
        return;
    }
    public static void rbfsetalgoqnn(rbfmodel s)
    {
        double q;
        double z;


        q = 1.0;
        z = 5.0;
        rbf.rbfsetalgoqnn(s.innerobj, q, z);

        return;
    }

    /*************************************************************************
    This  function  sets  RBF interpolation algorithm. ALGLIB supports several
    RBF algorithms with different properties.

    This  algorithm is called RBF-ML. It builds  multilayer  RBF  model,  i.e.
    model with subsequently decreasing  radii,  which  allows  us  to  combine
    smoothness (due to  large radii of  the first layers) with  exactness (due
    to small radii of the last layers) and fast convergence.

    Internally RBF-ML uses many different  means  of acceleration, from sparse
    matrices  to  KD-trees,  which  results in algorithm whose working time is
    roughly proportional to N*log(N)*Density*RBase^2*NLayers,  where  N  is  a
    number of points, Density is an average density if points per unit of  the
    interpolation space, RBase is an initial radius, NLayers is  a  number  of
    layers.

    RBF-ML is good for following kinds of interpolation problems:
    1. "exact" problems (perfect fit) with well separated points
    2. least squares problems with arbitrary distribution of points (algorithm
       gives  perfect  fit  where it is possible, and resorts to least squares
       fit in the hard areas).
    3. noisy problems where  we  want  to  apply  some  controlled  amount  of
       smoothing.

    INPUT PARAMETERS:
        S       -   RBF model, initialized by RBFCreate() call
        RBase   -   RBase parameter, RBase>0
        NLayers -   NLayers parameter, NLayers>0, recommended value  to  start
                    with - about 5.
        LambdaV -   regularization value, can be useful when  solving  problem
                    in the least squares sense.  Optimal  lambda  is  problem-
                    dependent and require trial and error. In our  experience,
                    good lambda can be as large as 0.1, and you can use  0.001
                    as initial guess.
                    Default  value  - 0.01, which is used when LambdaV is  not
                    given.  You  can  specify  zero  value,  but  it  is   not
                    recommended to do so.

    TUNING ALGORITHM

    In order to use this algorithm you have to choose three parameters:
    * initial radius RBase
    * number of layers in the model NLayers
    * regularization coefficient LambdaV

    Initial radius is easy to choose - you can pick any number  several  times
    larger  than  the  average  distance between points. Algorithm won't break
    down if you choose radius which is too large (model construction time will
    increase, but model will be built correctly).

    Choose such number of layers that RLast=RBase/2^(NLayers-1)  (radius  used
    by  the  last  layer)  will  be  smaller than the typical distance between
    points.  In  case  model  error  is  too large, you can increase number of
    layers.  Having  more  layers  will make model construction and evaluation
    proportionally slower, but it will allow you to have model which precisely
    fits your data. From the other side, if you want to  suppress  noise,  you
    can DECREASE number of layers to make your model less flexible.

    Regularization coefficient LambdaV controls smoothness of  the  individual
    models built for each layer. We recommend you to use default value in case
    you don't want to tune this parameter,  because  having  non-zero  LambdaV
    accelerates and stabilizes internal iterative algorithm. In case you  want
    to suppress noise you can use  LambdaV  as  additional  parameter  (larger
    value = more smoothness) to tune.

    TYPICAL ERRORS

    1. Using  initial  radius  which is too large. Memory requirements  of the
       RBF-ML are roughly proportional to N*Density*RBase^2 (where Density  is
       an average density of points per unit of the interpolation  space).  In
       the extreme case of the very large RBase we will need O(N^2)  units  of
       memory - and many layers in order to decrease radius to some reasonably
       small value.

    2. Using too small number of layers - RBF models with large radius are not
       flexible enough to reproduce small variations in the  target  function.
       You  need  many  layers  with  different radii, from large to small, in
       order to have good model.

    3. Using  initial  radius  which  is  too  small.  You will get model with
       "holes" in the areas which are too far away from interpolation centers.
       However, algorithm will work correctly (and quickly) in this case.

    4. Using too many layers - you will get too large and too slow model. This
       model  will  perfectly  reproduce  your function, but maybe you will be
       able to achieve similar results with less layers (and less memory).

      -- ALGLIB --
         Copyright 02.03.2012 by Bochkanov Sergey
    *************************************************************************/
    public static void rbfsetalgomultilayer(rbfmodel s, double rbase, int nlayers, double lambdav)
    {

        rbf.rbfsetalgomultilayer(s.innerobj, rbase, nlayers, lambdav);
        return;
    }
    public static void rbfsetalgomultilayer(rbfmodel s, double rbase, int nlayers)
    {
        double lambdav;


        lambdav = 0.01;
        rbf.rbfsetalgomultilayer(s.innerobj, rbase, nlayers, lambdav);

        return;
    }

    /*************************************************************************
    This function sets linear term (model is a sum of radial  basis  functions
    plus linear polynomial). This function won't have effect until  next  call
    to RBFBuildModel().

    INPUT PARAMETERS:
        S       -   RBF model, initialized by RBFCreate() call

    NOTE: this   function  has   some   serialization-related  subtleties.  We
          recommend you to study serialization examples from ALGLIB  Reference
          Manual if you want to perform serialization of your models.

      -- ALGLIB --
         Copyright 13.12.2011 by Bochkanov Sergey
    *************************************************************************/
    public static void rbfsetlinterm(rbfmodel s)
    {

        rbf.rbfsetlinterm(s.innerobj);
        return;
    }

    /*************************************************************************
    This function sets constant term (model is a sum of radial basis functions
    plus constant).  This  function  won't  have  effect  until  next  call to
    RBFBuildModel().

    INPUT PARAMETERS:
        S       -   RBF model, initialized by RBFCreate() call

    NOTE: this   function  has   some   serialization-related  subtleties.  We
          recommend you to study serialization examples from ALGLIB  Reference
          Manual if you want to perform serialization of your models.

      -- ALGLIB --
         Copyright 13.12.2011 by Bochkanov Sergey
    *************************************************************************/
    public static void rbfsetconstterm(rbfmodel s)
    {

        rbf.rbfsetconstterm(s.innerobj);
        return;
    }

    /*************************************************************************
    This  function  sets  zero  term (model is a sum of radial basis functions
    without polynomial term). This function won't have effect until next  call
    to RBFBuildModel().

    INPUT PARAMETERS:
        S       -   RBF model, initialized by RBFCreate() call

    NOTE: this   function  has   some   serialization-related  subtleties.  We
          recommend you to study serialization examples from ALGLIB  Reference
          Manual if you want to perform serialization of your models.

      -- ALGLIB --
         Copyright 13.12.2011 by Bochkanov Sergey
    *************************************************************************/
    public static void rbfsetzeroterm(rbfmodel s)
    {

        rbf.rbfsetzeroterm(s.innerobj);
        return;
    }

    /*************************************************************************
    This   function  builds  RBF  model  and  returns  report  (contains  some
    information which can be used for evaluation of the algorithm properties).

    Call to this function modifies RBF model by calculating its centers/radii/
    weights  and  saving  them  into  RBFModel  structure.  Initially RBFModel
    contain zero coefficients, but after call to this function  we  will  have
    coefficients which were calculated in order to fit our dataset.

    After you called this function you can call RBFCalc(),  RBFGridCalc()  and
    other model calculation functions.

    INPUT PARAMETERS:
        S       -   RBF model, initialized by RBFCreate() call
        Rep     -   report:
                    * Rep.TerminationType:
                      * -5 - non-distinct basis function centers were detected,
                             interpolation aborted
                      * -4 - nonconvergence of the internal SVD solver
                      *  1 - successful termination
                    Fields are used for debugging purposes:
                    * Rep.IterationsCount - iterations count of the LSQR solver
                    * Rep.NMV - number of matrix-vector products
                    * Rep.ARows - rows count for the system matrix
                    * Rep.ACols - columns count for the system matrix
                    * Rep.ANNZ - number of significantly non-zero elements
                      (elements above some algorithm-determined threshold)

    NOTE:  failure  to  build  model will leave current state of the structure
    unchanged.

      -- ALGLIB --
         Copyright 13.12.2011 by Bochkanov Sergey
    *************************************************************************/
    public static void rbfbuildmodel(rbfmodel s, out rbfreport rep)
    {
        rep = new rbfreport();
        rbf.rbfbuildmodel(s.innerobj, rep.innerobj);
        return;
    }

    /*************************************************************************
    This function calculates values of the RBF model in the given point.

    This function should be used when we have NY=1 (scalar function) and  NX=2
    (2-dimensional space). If you have 3-dimensional space, use RBFCalc3(). If
    you have general situation (NX-dimensional space, NY-dimensional function)
    you should use general, less efficient implementation RBFCalc().

    If  you  want  to  calculate  function  values  many times, consider using
    RBFGridCalc2(), which is far more efficient than many subsequent calls  to
    RBFCalc2().

    This function returns 0.0 when:
    * model is not initialized
    * NX<>2
     *NY<>1

    INPUT PARAMETERS:
        S       -   RBF model
        X0      -   first coordinate, finite number
        X1      -   second coordinate, finite number

    RESULT:
        value of the model or 0.0 (as defined above)

      -- ALGLIB --
         Copyright 13.12.2011 by Bochkanov Sergey
    *************************************************************************/
    public static double rbfcalc2(rbfmodel s, double x0, double x1)
    {

        double result = rbf.rbfcalc2(s.innerobj, x0, x1);
        return result;
    }

    /*************************************************************************
    This function calculates values of the RBF model in the given point.

    This function should be used when we have NY=1 (scalar function) and  NX=3
    (3-dimensional space). If you have 2-dimensional space, use RBFCalc2(). If
    you have general situation (NX-dimensional space, NY-dimensional function)
    you should use general, less efficient implementation RBFCalc().

    This function returns 0.0 when:
    * model is not initialized
    * NX<>3
     *NY<>1

    INPUT PARAMETERS:
        S       -   RBF model
        X0      -   first coordinate, finite number
        X1      -   second coordinate, finite number
        X2      -   third coordinate, finite number

    RESULT:
        value of the model or 0.0 (as defined above)

      -- ALGLIB --
         Copyright 13.12.2011 by Bochkanov Sergey
    *************************************************************************/
    public static double rbfcalc3(rbfmodel s, double x0, double x1, double x2)
    {

        double result = rbf.rbfcalc3(s.innerobj, x0, x1, x2);
        return result;
    }

    /*************************************************************************
    This function calculates values of the RBF model at the given point.

    This is general function which can be used for arbitrary NX (dimension  of
    the space of arguments) and NY (dimension of the function itself). However
    when  you  have  NY=1  you  may  find more convenient to use RBFCalc2() or
    RBFCalc3().

    This function returns 0.0 when model is not initialized.

    INPUT PARAMETERS:
        S       -   RBF model
        X       -   coordinates, array[NX].
                    X may have more than NX elements, in this case only
                    leading NX will be used.

    OUTPUT PARAMETERS:
        Y       -   function value, array[NY]. Y is out-parameter and
                    reallocated after call to this function. In case you  want
                    to reuse previously allocated Y, you may use RBFCalcBuf(),
                    which reallocates Y only when it is too small.

      -- ALGLIB --
         Copyright 13.12.2011 by Bochkanov Sergey
    *************************************************************************/
    public static void rbfcalc(rbfmodel s, double[] x, out double[] y)
    {
        y = new double[0];
        rbf.rbfcalc(s.innerobj, x, ref y);
        return;
    }

    /*************************************************************************
    This function calculates values of the RBF model at the given point.

    Same as RBFCalc(), but does not reallocate Y when in is large enough to
    store function values.

    INPUT PARAMETERS:
        S       -   RBF model
        X       -   coordinates, array[NX].
                    X may have more than NX elements, in this case only
                    leading NX will be used.
        Y       -   possibly preallocated array

    OUTPUT PARAMETERS:
        Y       -   function value, array[NY]. Y is not reallocated when it
                    is larger than NY.

      -- ALGLIB --
         Copyright 13.12.2011 by Bochkanov Sergey
    *************************************************************************/
    public static void rbfcalcbuf(rbfmodel s, double[] x, ref double[] y)
    {

        rbf.rbfcalcbuf(s.innerobj, x, ref y);
        return;
    }

    /*************************************************************************
    This function calculates values of the RBF model at the regular grid.

    Grid have N0*N1 points, with Point[I,J] = (X0[I], X1[J])

    This function returns 0.0 when:
    * model is not initialized
    * NX<>2
     *NY<>1

    INPUT PARAMETERS:
        S       -   RBF model
        X0      -   array of grid nodes, first coordinates, array[N0]
        N0      -   grid size (number of nodes) in the first dimension
        X1      -   array of grid nodes, second coordinates, array[N1]
        N1      -   grid size (number of nodes) in the second dimension

    OUTPUT PARAMETERS:
        Y       -   function values, array[N0,N1]. Y is out-variable and
                    is reallocated by this function.

      -- ALGLIB --
         Copyright 13.12.2011 by Bochkanov Sergey
    *************************************************************************/
    public static void rbfgridcalc2(rbfmodel s, double[] x0, int n0, double[] x1, int n1, out double[,] y)
    {
        y = new double[0,0];
        rbf.rbfgridcalc2(s.innerobj, x0, n0, x1, n1, ref y);
        return;
    }

    /*************************************************************************
    This function "unpacks" RBF model by extracting its coefficients.

    INPUT PARAMETERS:
        S       -   RBF model

    OUTPUT PARAMETERS:
        NX      -   dimensionality of argument
        NY      -   dimensionality of the target function
        XWR     -   model information, array[NC,NX+NY+1].
                    One row of the array corresponds to one basis function:
                    * first NX columns  - coordinates of the center
                    * next NY columns   - weights, one per dimension of the
                                          function being modelled
                    * last column       - radius, same for all dimensions of
                                          the function being modelled
        NC      -   number of the centers
        V       -   polynomial  term , array[NY,NX+1]. One row per one
                    dimension of the function being modelled. First NX
                    elements are linear coefficients, V[NX] is equal to the
                    constant part.

      -- ALGLIB --
         Copyright 13.12.2011 by Bochkanov Sergey
    *************************************************************************/
    public static void rbfunpack(rbfmodel s, out int nx, out int ny, out double[,] xwr, out int nc, out double[,] v)
    {
        nx = 0;
        ny = 0;
        xwr = new double[0,0];
        nc = 0;
        v = new double[0,0];
        rbf.rbfunpack(s.innerobj, ref nx, ref ny, ref xwr, ref nc, ref v);
        return;
    }

}
public partial class alglib
{


    /*************************************************************************
    2-dimensional spline inteprolant
    *************************************************************************/
    public class spline2dinterpolant
    {
        //
        // Public declarations
        //

        public spline2dinterpolant()
        {
            _innerobj = new spline2d.spline2dinterpolant();
        }

        //
        // Although some of declarations below are public, you should not use them
        // They are intended for internal use only
        //
        private spline2d.spline2dinterpolant _innerobj;
        public spline2d.spline2dinterpolant innerobj { get { return _innerobj; } }
        public spline2dinterpolant(spline2d.spline2dinterpolant obj)
        {
            _innerobj = obj;
        }
    }

    /*************************************************************************
    This subroutine builds bilinear spline coefficients table.

    Input parameters:
        X   -   spline abscissas, array[0..N-1]
        Y   -   spline ordinates, array[0..M-1]
        F   -   function values, array[0..M-1,0..N-1]
        M,N -   grid size, M>=2, N>=2

    Output parameters:
        C   -   spline interpolant

      -- ALGLIB PROJECT --
         Copyright 05.07.2007 by Bochkanov Sergey
    *************************************************************************/
    public static void spline2dbuildbilinear(double[] x, double[] y, double[,] f, int m, int n, out spline2dinterpolant c)
    {
        c = new spline2dinterpolant();
        spline2d.spline2dbuildbilinear(x, y, f, m, n, c.innerobj);
        return;
    }

    /*************************************************************************
    This subroutine builds bicubic spline coefficients table.

    Input parameters:
        X   -   spline abscissas, array[0..N-1]
        Y   -   spline ordinates, array[0..M-1]
        F   -   function values, array[0..M-1,0..N-1]
        M,N -   grid size, M>=2, N>=2

    Output parameters:
        C   -   spline interpolant

      -- ALGLIB PROJECT --
         Copyright 05.07.2007 by Bochkanov Sergey
    *************************************************************************/
    public static void spline2dbuildbicubic(double[] x, double[] y, double[,] f, int m, int n, out spline2dinterpolant c)
    {
        c = new spline2dinterpolant();
        spline2d.spline2dbuildbicubic(x, y, f, m, n, c.innerobj);
        return;
    }

    /*************************************************************************
    This subroutine calculates the value of the bilinear or bicubic spline  at
    the given point X.

    Input parameters:
        C   -   coefficients table.
                Built by BuildBilinearSpline or BuildBicubicSpline.
        X, Y-   point

    Result:
        S(x,y)

      -- ALGLIB PROJECT --
         Copyright 05.07.2007 by Bochkanov Sergey
    *************************************************************************/
    public static double spline2dcalc(spline2dinterpolant c, double x, double y)
    {

        double result = spline2d.spline2dcalc(c.innerobj, x, y);
        return result;
    }

    /*************************************************************************
    This subroutine calculates the value of the bilinear or bicubic spline  at
    the given point X and its derivatives.

    Input parameters:
        C   -   spline interpolant.
        X, Y-   point

    Output parameters:
        F   -   S(x,y)
        FX  -   dS(x,y)/dX
        FY  -   dS(x,y)/dY
        FXY -   d2S(x,y)/dXdY

      -- ALGLIB PROJECT --
         Copyright 05.07.2007 by Bochkanov Sergey
    *************************************************************************/
    public static void spline2ddiff(spline2dinterpolant c, double x, double y, out double f, out double fx, out double fy, out double fxy)
    {
        f = 0;
        fx = 0;
        fy = 0;
        fxy = 0;
        spline2d.spline2ddiff(c.innerobj, x, y, ref f, ref fx, ref fy, ref fxy);
        return;
    }

    /*************************************************************************
    This subroutine unpacks two-dimensional spline into the coefficients table

    Input parameters:
        C   -   spline interpolant.

    Result:
        M, N-   grid size (x-axis and y-axis)
        Tbl -   coefficients table, unpacked format,
                [0..(N-1)*(M-1)-1, 0..19].
                For I = 0...M-2, J=0..N-2:
                    K =  I*(N-1)+J
                    Tbl[K,0] = X[j]
                    Tbl[K,1] = X[j+1]
                    Tbl[K,2] = Y[i]
                    Tbl[K,3] = Y[i+1]
                    Tbl[K,4] = C00
                    Tbl[K,5] = C01
                    Tbl[K,6] = C02
                    Tbl[K,7] = C03
                    Tbl[K,8] = C10
                    Tbl[K,9] = C11
                    ...
                    Tbl[K,19] = C33
                On each grid square spline is equals to:
                    S(x) = SUM(c[i,j]*(x^i)*(y^j), i=0..3, j=0..3)
                    t = x-x[j]
                    u = y-y[i]

      -- ALGLIB PROJECT --
         Copyright 29.06.2007 by Bochkanov Sergey
    *************************************************************************/
    public static void spline2dunpack(spline2dinterpolant c, out int m, out int n, out double[,] tbl)
    {
        m = 0;
        n = 0;
        tbl = new double[0,0];
        spline2d.spline2dunpack(c.innerobj, ref m, ref n, ref tbl);
        return;
    }

    /*************************************************************************
    This subroutine performs linear transformation of the spline argument.

    Input parameters:
        C       -   spline interpolant
        AX, BX  -   transformation coefficients: x = A*t + B
        AY, BY  -   transformation coefficients: y = A*u + B
    Result:
        C   -   transformed spline

      -- ALGLIB PROJECT --
         Copyright 30.06.2007 by Bochkanov Sergey
    *************************************************************************/
    public static void spline2dlintransxy(spline2dinterpolant c, double ax, double bx, double ay, double by)
    {

        spline2d.spline2dlintransxy(c.innerobj, ax, bx, ay, by);
        return;
    }

    /*************************************************************************
    This subroutine performs linear transformation of the spline.

    Input parameters:
        C   -   spline interpolant.
        A, B-   transformation coefficients: S2(x,y) = A*S(x,y) + B

    Output parameters:
        C   -   transformed spline

      -- ALGLIB PROJECT --
         Copyright 30.06.2007 by Bochkanov Sergey
    *************************************************************************/
    public static void spline2dlintransf(spline2dinterpolant c, double a, double b)
    {

        spline2d.spline2dlintransf(c.innerobj, a, b);
        return;
    }

    /*************************************************************************
    Bicubic spline resampling

    Input parameters:
        A           -   function values at the old grid,
                        array[0..OldHeight-1, 0..OldWidth-1]
        OldHeight   -   old grid height, OldHeight>1
        OldWidth    -   old grid width, OldWidth>1
        NewHeight   -   new grid height, NewHeight>1
        NewWidth    -   new grid width, NewWidth>1

    Output parameters:
        B           -   function values at the new grid,
                        array[0..NewHeight-1, 0..NewWidth-1]

      -- ALGLIB routine --
         15 May, 2007
         Copyright by Bochkanov Sergey
    *************************************************************************/
    public static void spline2dresamplebicubic(double[,] a, int oldheight, int oldwidth, out double[,] b, int newheight, int newwidth)
    {
        b = new double[0,0];
        spline2d.spline2dresamplebicubic(a, oldheight, oldwidth, ref b, newheight, newwidth);
        return;
    }

    /*************************************************************************
    Bilinear spline resampling

    Input parameters:
        A           -   function values at the old grid,
                        array[0..OldHeight-1, 0..OldWidth-1]
        OldHeight   -   old grid height, OldHeight>1
        OldWidth    -   old grid width, OldWidth>1
        NewHeight   -   new grid height, NewHeight>1
        NewWidth    -   new grid width, NewWidth>1

    Output parameters:
        B           -   function values at the new grid,
                        array[0..NewHeight-1, 0..NewWidth-1]

      -- ALGLIB routine --
         09.07.2007
         Copyright by Bochkanov Sergey
    *************************************************************************/
    public static void spline2dresamplebilinear(double[,] a, int oldheight, int oldwidth, out double[,] b, int newheight, int newwidth)
    {
        b = new double[0,0];
        spline2d.spline2dresamplebilinear(a, oldheight, oldwidth, ref b, newheight, newwidth);
        return;
    }

}
public partial class alglib
{
    public class idwint
    {
        /*************************************************************************
        IDW interpolant.
        *************************************************************************/
        public class idwinterpolant
        {
            public int n;
            public int nx;
            public int d;
            public double r;
            public int nw;
            public nearestneighbor.kdtree tree;
            public int modeltype;
            public double[,] q;
            public double[] xbuf;
            public int[] tbuf;
            public double[] rbuf;
            public double[,] xybuf;
            public int debugsolverfailures;
            public double debugworstrcond;
            public double debugbestrcond;
            public idwinterpolant()
            {
                tree = new nearestneighbor.kdtree();
                q = new double[0,0];
                xbuf = new double[0];
                tbuf = new int[0];
                rbuf = new double[0];
                xybuf = new double[0,0];
            }
        };




        public const double idwqfactor = 1.5;
        public const int idwkmin = 5;


        /*************************************************************************
        IDW interpolation

        INPUT PARAMETERS:
            Z   -   IDW interpolant built with one of model building
                    subroutines.
            X   -   array[0..NX-1], interpolation point

        Result:
            IDW interpolant Z(X)

          -- ALGLIB --
             Copyright 02.03.2010 by Bochkanov Sergey
        *************************************************************************/
        public static double idwcalc(idwinterpolant z,
            double[] x)
        {
            double result = 0;
            int nx = 0;
            int i = 0;
            int k = 0;
            double r = 0;
            double s = 0;
            double w = 0;
            double v1 = 0;
            double v2 = 0;
            double d0 = 0;
            double di = 0;

            
            //
            // these initializers are not really necessary,
            // but without them compiler complains about uninitialized locals
            //
            k = 0;
            
            //
            // Query
            //
            if( z.modeltype==0 )
            {
                
                //
                // NQ/NW-based model
                //
                nx = z.nx;
                k = nearestneighbor.kdtreequeryknn(z.tree, x, z.nw, true);
                nearestneighbor.kdtreequeryresultsdistances(z.tree, ref z.rbuf);
                nearestneighbor.kdtreequeryresultstags(z.tree, ref z.tbuf);
            }
            if( z.modeltype==1 )
            {
                
                //
                // R-based model
                //
                nx = z.nx;
                k = nearestneighbor.kdtreequeryrnn(z.tree, x, z.r, true);
                nearestneighbor.kdtreequeryresultsdistances(z.tree, ref z.rbuf);
                nearestneighbor.kdtreequeryresultstags(z.tree, ref z.tbuf);
                if( k<idwkmin )
                {
                    
                    //
                    // we need at least IDWKMin points
                    //
                    k = nearestneighbor.kdtreequeryknn(z.tree, x, idwkmin, true);
                    nearestneighbor.kdtreequeryresultsdistances(z.tree, ref z.rbuf);
                    nearestneighbor.kdtreequeryresultstags(z.tree, ref z.tbuf);
                }
            }
            
            //
            // initialize weights for linear/quadratic members calculation.
            //
            // NOTE 1: weights are calculated using NORMALIZED modified
            // Shepard's formula. Original formula gives w(i) = sqr((R-di)/(R*di)),
            // where di is i-th distance, R is max(di). Modified formula have
            // following form:
            //     w_mod(i) = 1, if di=d0
            //     w_mod(i) = w(i)/w(0), if di<>d0
            //
            // NOTE 2: self-match is USED for this query
            //
            // NOTE 3: last point almost always gain zero weight, but it MUST
            // be used for fitting because sometimes it will gain NON-ZERO
            // weight - for example, when all distances are equal.
            //
            r = z.rbuf[k-1];
            d0 = z.rbuf[0];
            result = 0;
            s = 0;
            for(i=0; i<=k-1; i++)
            {
                di = z.rbuf[i];
                if( (double)(di)==(double)(d0) )
                {
                    
                    //
                    // distance is equal to shortest, set it 1.0
                    // without explicitly calculating (which would give
                    // us same result, but 'll expose us to the risk of
                    // division by zero).
                    //
                    w = 1;
                }
                else
                {
                    
                    //
                    // use normalized formula
                    //
                    v1 = (r-di)/(r-d0);
                    v2 = d0/di;
                    w = math.sqr(v1*v2);
                }
                result = result+w*idwcalcq(z, x, z.tbuf[i]);
                s = s+w;
            }
            result = result/s;
            return result;
        }


        /*************************************************************************
        IDW interpolant using modified Shepard method for uniform point
        distributions.

        INPUT PARAMETERS:
            XY  -   X and Y values, array[0..N-1,0..NX].
                    First NX columns contain X-values, last column contain
                    Y-values.
            N   -   number of nodes, N>0.
            NX  -   space dimension, NX>=1.
            D   -   nodal function type, either:
                    * 0     constant  model.  Just  for  demonstration only, worst
                            model ever.
                    * 1     linear model, least squares fitting. Simpe  model  for
                            datasets too small for quadratic models
                    * 2     quadratic  model,  least  squares  fitting. Best model
                            available (if your dataset is large enough).
                    * -1    "fast"  linear  model,  use  with  caution!!!   It  is
                            significantly  faster than linear/quadratic and better
                            than constant model. But it is less robust (especially
                            in the presence of noise).
            NQ  -   number of points used to calculate  nodal  functions  (ignored
                    for constant models). NQ should be LARGER than:
                    * max(1.5*(1+NX),2^NX+1) for linear model,
                    * max(3/4*(NX+2)*(NX+1),2^NX+1) for quadratic model.
                    Values less than this threshold will be silently increased.
            NW  -   number of points used to calculate weights and to interpolate.
                    Required: >=2^NX+1, values less than this  threshold  will  be
                    silently increased.
                    Recommended value: about 2*NQ

        OUTPUT PARAMETERS:
            Z   -   IDW interpolant.
            
        NOTES:
          * best results are obtained with quadratic models, worst - with constant
            models
          * when N is large, NQ and NW must be significantly smaller than  N  both
            to obtain optimal performance and to obtain optimal accuracy. In 2  or
            3-dimensional tasks NQ=15 and NW=25 are good values to start with.
          * NQ  and  NW  may  be  greater  than  N.  In  such  cases  they will be
            automatically decreased.
          * this subroutine is always succeeds (as long as correct parameters  are
            passed).
          * see  'Multivariate  Interpolation  of Large Sets of Scattered Data' by
            Robert J. Renka for more information on this algorithm.
          * this subroutine assumes that point distribution is uniform at the small
            scales.  If  it  isn't  -  for  example,  points are concentrated along
            "lines", but "lines" distribution is uniform at the larger scale - then
            you should use IDWBuildModifiedShepardR()


          -- ALGLIB PROJECT --
             Copyright 02.03.2010 by Bochkanov Sergey
        *************************************************************************/
        public static void idwbuildmodifiedshepard(double[,] xy,
            int n,
            int nx,
            int d,
            int nq,
            int nw,
            idwinterpolant z)
        {
            int i = 0;
            int j = 0;
            int k = 0;
            int j2 = 0;
            int j3 = 0;
            double v = 0;
            double r = 0;
            double s = 0;
            double d0 = 0;
            double di = 0;
            double v1 = 0;
            double v2 = 0;
            int nc = 0;
            int offs = 0;
            double[] x = new double[0];
            double[] qrbuf = new double[0];
            double[,] qxybuf = new double[0,0];
            double[] y = new double[0];
            double[,] fmatrix = new double[0,0];
            double[] w = new double[0];
            double[] qsol = new double[0];
            double[] temp = new double[0];
            int[] tags = new int[0];
            int info = 0;
            double taskrcond = 0;
            int i_ = 0;

            
            //
            // these initializers are not really necessary,
            // but without them compiler complains about uninitialized locals
            //
            nc = 0;
            
            //
            // assertions
            //
            alglib.ap.assert(n>0, "IDWBuildModifiedShepard: N<=0!");
            alglib.ap.assert(nx>=1, "IDWBuildModifiedShepard: NX<1!");
            alglib.ap.assert(d>=-1 && d<=2, "IDWBuildModifiedShepard: D<>-1 and D<>0 and D<>1 and D<>2!");
            
            //
            // Correct parameters if needed
            //
            if( d==1 )
            {
                nq = Math.Max(nq, (int)Math.Ceiling(idwqfactor*(1+nx))+1);
                nq = Math.Max(nq, (int)Math.Round(Math.Pow(2, nx))+1);
            }
            if( d==2 )
            {
                nq = Math.Max(nq, (int)Math.Ceiling(idwqfactor*(nx+2)*(nx+1)/2)+1);
                nq = Math.Max(nq, (int)Math.Round(Math.Pow(2, nx))+1);
            }
            nw = Math.Max(nw, (int)Math.Round(Math.Pow(2, nx))+1);
            nq = Math.Min(nq, n);
            nw = Math.Min(nw, n);
            
            //
            // primary initialization of Z
            //
            idwinit1(n, nx, d, nq, nw, z);
            z.modeltype = 0;
            
            //
            // Create KD-tree
            //
            tags = new int[n];
            for(i=0; i<=n-1; i++)
            {
                tags[i] = i;
            }
            nearestneighbor.kdtreebuildtagged(xy, tags, n, nx, 1, 2, z.tree);
            
            //
            // build nodal functions
            //
            temp = new double[nq+1];
            x = new double[nx];
            qrbuf = new double[nq];
            qxybuf = new double[nq, nx+1];
            if( d==-1 )
            {
                w = new double[nq];
            }
            if( d==1 )
            {
                y = new double[nq];
                w = new double[nq];
                qsol = new double[nx];
                
                //
                // NX for linear members,
                // 1 for temporary storage
                //
                fmatrix = new double[nq, nx+1];
            }
            if( d==2 )
            {
                y = new double[nq];
                w = new double[nq];
                qsol = new double[nx+(int)Math.Round(nx*(nx+1)*0.5)];
                
                //
                // NX for linear members,
                // Round(NX*(NX+1)*0.5) for quadratic model,
                // 1 for temporary storage
                //
                fmatrix = new double[nq, nx+(int)Math.Round(nx*(nx+1)*0.5)+1];
            }
            for(i=0; i<=n-1; i++)
            {
                
                //
                // Initialize center and function value.
                // If D=0 it is all what we need
                //
                for(i_=0; i_<=nx;i_++)
                {
                    z.q[i,i_] = xy[i,i_];
                }
                if( d==0 )
                {
                    continue;
                }
                
                //
                // calculate weights for linear/quadratic members calculation.
                //
                // NOTE 1: weights are calculated using NORMALIZED modified
                // Shepard's formula. Original formula is w(i) = sqr((R-di)/(R*di)),
                // where di is i-th distance, R is max(di). Modified formula have
                // following form:
                //     w_mod(i) = 1, if di=d0
                //     w_mod(i) = w(i)/w(0), if di<>d0
                //
                // NOTE 2: self-match is NOT used for this query
                //
                // NOTE 3: last point almost always gain zero weight, but it MUST
                // be used for fitting because sometimes it will gain NON-ZERO
                // weight - for example, when all distances are equal.
                //
                for(i_=0; i_<=nx-1;i_++)
                {
                    x[i_] = xy[i,i_];
                }
                k = nearestneighbor.kdtreequeryknn(z.tree, x, nq, false);
                nearestneighbor.kdtreequeryresultsxy(z.tree, ref qxybuf);
                nearestneighbor.kdtreequeryresultsdistances(z.tree, ref qrbuf);
                r = qrbuf[k-1];
                d0 = qrbuf[0];
                for(j=0; j<=k-1; j++)
                {
                    di = qrbuf[j];
                    if( (double)(di)==(double)(d0) )
                    {
                        
                        //
                        // distance is equal to shortest, set it 1.0
                        // without explicitly calculating (which would give
                        // us same result, but 'll expose us to the risk of
                        // division by zero).
                        //
                        w[j] = 1;
                    }
                    else
                    {
                        
                        //
                        // use normalized formula
                        //
                        v1 = (r-di)/(r-d0);
                        v2 = d0/di;
                        w[j] = math.sqr(v1*v2);
                    }
                }
                
                //
                // calculate linear/quadratic members
                //
                if( d==-1 )
                {
                    
                    //
                    // "Fast" linear nodal function calculated using
                    // inverse distance weighting
                    //
                    for(j=0; j<=nx-1; j++)
                    {
                        x[j] = 0;
                    }
                    s = 0;
                    for(j=0; j<=k-1; j++)
                    {
                        
                        //
                        // calculate J-th inverse distance weighted gradient:
                        //     grad_k = (y_j-y_k)*(x_j-x_k)/sqr(norm(x_j-x_k))
                        //     grad   = sum(wk*grad_k)/sum(w_k)
                        //
                        v = 0;
                        for(j2=0; j2<=nx-1; j2++)
                        {
                            v = v+math.sqr(qxybuf[j,j2]-xy[i,j2]);
                        }
                        
                        //
                        // Although x_j<>x_k, sqr(norm(x_j-x_k)) may be zero due to
                        // underflow. If it is, we assume than J-th gradient is zero
                        // (i.e. don't add anything)
                        //
                        if( (double)(v)!=(double)(0) )
                        {
                            for(j2=0; j2<=nx-1; j2++)
                            {
                                x[j2] = x[j2]+w[j]*(qxybuf[j,nx]-xy[i,nx])*(qxybuf[j,j2]-xy[i,j2])/v;
                            }
                        }
                        s = s+w[j];
                    }
                    for(j=0; j<=nx-1; j++)
                    {
                        z.q[i,nx+1+j] = x[j]/s;
                    }
                }
                else
                {
                    
                    //
                    // Least squares models: build
                    //
                    if( d==1 )
                    {
                        
                        //
                        // Linear nodal function calculated using
                        // least squares fitting to its neighbors
                        //
                        for(j=0; j<=k-1; j++)
                        {
                            for(j2=0; j2<=nx-1; j2++)
                            {
                                fmatrix[j,j2] = qxybuf[j,j2]-xy[i,j2];
                            }
                            y[j] = qxybuf[j,nx]-xy[i,nx];
                        }
                        nc = nx;
                    }
                    if( d==2 )
                    {
                        
                        //
                        // Quadratic nodal function calculated using
                        // least squares fitting to its neighbors
                        //
                        for(j=0; j<=k-1; j++)
                        {
                            offs = 0;
                            for(j2=0; j2<=nx-1; j2++)
                            {
                                fmatrix[j,offs] = qxybuf[j,j2]-xy[i,j2];
                                offs = offs+1;
                            }
                            for(j2=0; j2<=nx-1; j2++)
                            {
                                for(j3=j2; j3<=nx-1; j3++)
                                {
                                    fmatrix[j,offs] = (qxybuf[j,j2]-xy[i,j2])*(qxybuf[j,j3]-xy[i,j3]);
                                    offs = offs+1;
                                }
                            }
                            y[j] = qxybuf[j,nx]-xy[i,nx];
                        }
                        nc = nx+(int)Math.Round(nx*(nx+1)*0.5);
                    }
                    idwinternalsolver(ref y, ref w, ref fmatrix, ref temp, k, nc, ref info, ref qsol, ref taskrcond);
                    
                    //
                    // Least squares models: copy results
                    //
                    if( info>0 )
                    {
                        
                        //
                        // LLS task is solved, copy results
                        //
                        z.debugworstrcond = Math.Min(z.debugworstrcond, taskrcond);
                        z.debugbestrcond = Math.Max(z.debugbestrcond, taskrcond);
                        for(j=0; j<=nc-1; j++)
                        {
                            z.q[i,nx+1+j] = qsol[j];
                        }
                    }
                    else
                    {
                        
                        //
                        // Solver failure, very strange, but we will use
                        // zero values to handle it.
                        //
                        z.debugsolverfailures = z.debugsolverfailures+1;
                        for(j=0; j<=nc-1; j++)
                        {
                            z.q[i,nx+1+j] = 0;
                        }
                    }
                }
            }
        }


        /*************************************************************************
        IDW interpolant using modified Shepard method for non-uniform datasets.

        This type of model uses  constant  nodal  functions and interpolates using
        all nodes which are closer than user-specified radius R. It  may  be  used
        when points distribution is non-uniform at the small scale, but it  is  at
        the distances as large as R.

        INPUT PARAMETERS:
            XY  -   X and Y values, array[0..N-1,0..NX].
                    First NX columns contain X-values, last column contain
                    Y-values.
            N   -   number of nodes, N>0.
            NX  -   space dimension, NX>=1.
            R   -   radius, R>0

        OUTPUT PARAMETERS:
            Z   -   IDW interpolant.

        NOTES:
        * if there is less than IDWKMin points within  R-ball,  algorithm  selects
          IDWKMin closest ones, so that continuity properties of  interpolant  are
          preserved even far from points.

          -- ALGLIB PROJECT --
             Copyright 11.04.2010 by Bochkanov Sergey
        *************************************************************************/
        public static void idwbuildmodifiedshepardr(double[,] xy,
            int n,
            int nx,
            double r,
            idwinterpolant z)
        {
            int i = 0;
            int[] tags = new int[0];
            int i_ = 0;

            
            //
            // assertions
            //
            alglib.ap.assert(n>0, "IDWBuildModifiedShepardR: N<=0!");
            alglib.ap.assert(nx>=1, "IDWBuildModifiedShepardR: NX<1!");
            alglib.ap.assert((double)(r)>(double)(0), "IDWBuildModifiedShepardR: R<=0!");
            
            //
            // primary initialization of Z
            //
            idwinit1(n, nx, 0, 0, n, z);
            z.modeltype = 1;
            z.r = r;
            
            //
            // Create KD-tree
            //
            tags = new int[n];
            for(i=0; i<=n-1; i++)
            {
                tags[i] = i;
            }
            nearestneighbor.kdtreebuildtagged(xy, tags, n, nx, 1, 2, z.tree);
            
            //
            // build nodal functions
            //
            for(i=0; i<=n-1; i++)
            {
                for(i_=0; i_<=nx;i_++)
                {
                    z.q[i,i_] = xy[i,i_];
                }
            }
        }


        /*************************************************************************
        IDW model for noisy data.

        This subroutine may be used to handle noisy data, i.e. data with noise  in
        OUTPUT values.  It differs from IDWBuildModifiedShepard() in the following
        aspects:
        * nodal functions are not constrained to pass through  nodes:  Qi(xi)<>yi,
          i.e. we have fitting  instead  of  interpolation.
        * weights which are used during least  squares fitting stage are all equal
          to 1.0 (independently of distance)
        * "fast"-linear or constant nodal functions are not supported (either  not
          robust enough or too rigid)

        This problem require far more complex tuning than interpolation  problems.
        Below you can find some recommendations regarding this problem:
        * focus on tuning NQ; it controls noise reduction. As for NW, you can just
          make it equal to 2*NQ.
        * you can use cross-validation to determine optimal NQ.
        * optimal NQ is a result of complex tradeoff  between  noise  level  (more
          noise = larger NQ required) and underlying  function  complexity  (given
          fixed N, larger NQ means smoothing of compex features in the data).  For
          example, NQ=N will reduce noise to the minimum level possible,  but  you
          will end up with just constant/linear/quadratic (depending on  D)  least
          squares model for the whole dataset.

        INPUT PARAMETERS:
            XY  -   X and Y values, array[0..N-1,0..NX].
                    First NX columns contain X-values, last column contain
                    Y-values.
            N   -   number of nodes, N>0.
            NX  -   space dimension, NX>=1.
            D   -   nodal function degree, either:
                    * 1     linear model, least squares fitting. Simpe  model  for
                            datasets too small for quadratic models (or  for  very
                            noisy problems).
                    * 2     quadratic  model,  least  squares  fitting. Best model
                            available (if your dataset is large enough).
            NQ  -   number of points used to calculate nodal functions.  NQ should
                    be  significantly   larger   than  1.5  times  the  number  of
                    coefficients in a nodal function to overcome effects of noise:
                    * larger than 1.5*(1+NX) for linear model,
                    * larger than 3/4*(NX+2)*(NX+1) for quadratic model.
                    Values less than this threshold will be silently increased.
            NW  -   number of points used to calculate weights and to interpolate.
                    Required: >=2^NX+1, values less than this  threshold  will  be
                    silently increased.
                    Recommended value: about 2*NQ or larger

        OUTPUT PARAMETERS:
            Z   -   IDW interpolant.

        NOTES:
          * best results are obtained with quadratic models, linear models are not
            recommended to use unless you are pretty sure that it is what you want
          * this subroutine is always succeeds (as long as correct parameters  are
            passed).
          * see  'Multivariate  Interpolation  of Large Sets of Scattered Data' by
            Robert J. Renka for more information on this algorithm.


          -- ALGLIB PROJECT --
             Copyright 02.03.2010 by Bochkanov Sergey
        *************************************************************************/
        public static void idwbuildnoisy(double[,] xy,
            int n,
            int nx,
            int d,
            int nq,
            int nw,
            idwinterpolant z)
        {
            int i = 0;
            int j = 0;
            int k = 0;
            int j2 = 0;
            int j3 = 0;
            double v = 0;
            int nc = 0;
            int offs = 0;
            double taskrcond = 0;
            double[] x = new double[0];
            double[] qrbuf = new double[0];
            double[,] qxybuf = new double[0,0];
            double[] y = new double[0];
            double[] w = new double[0];
            double[,] fmatrix = new double[0,0];
            double[] qsol = new double[0];
            int[] tags = new int[0];
            double[] temp = new double[0];
            int info = 0;
            int i_ = 0;

            
            //
            // these initializers are not really necessary,
            // but without them compiler complains about uninitialized locals
            //
            nc = 0;
            
            //
            // assertions
            //
            alglib.ap.assert(n>0, "IDWBuildNoisy: N<=0!");
            alglib.ap.assert(nx>=1, "IDWBuildNoisy: NX<1!");
            alglib.ap.assert(d>=1 && d<=2, "IDWBuildNoisy: D<>1 and D<>2!");
            
            //
            // Correct parameters if needed
            //
            if( d==1 )
            {
                nq = Math.Max(nq, (int)Math.Ceiling(idwqfactor*(1+nx))+1);
            }
            if( d==2 )
            {
                nq = Math.Max(nq, (int)Math.Ceiling(idwqfactor*(nx+2)*(nx+1)/2)+1);
            }
            nw = Math.Max(nw, (int)Math.Round(Math.Pow(2, nx))+1);
            nq = Math.Min(nq, n);
            nw = Math.Min(nw, n);
            
            //
            // primary initialization of Z
            //
            idwinit1(n, nx, d, nq, nw, z);
            z.modeltype = 0;
            
            //
            // Create KD-tree
            //
            tags = new int[n];
            for(i=0; i<=n-1; i++)
            {
                tags[i] = i;
            }
            nearestneighbor.kdtreebuildtagged(xy, tags, n, nx, 1, 2, z.tree);
            
            //
            // build nodal functions
            // (special algorithm for noisy data is used)
            //
            temp = new double[nq+1];
            x = new double[nx];
            qrbuf = new double[nq];
            qxybuf = new double[nq, nx+1];
            if( d==1 )
            {
                y = new double[nq];
                w = new double[nq];
                qsol = new double[1+nx];
                
                //
                // 1 for constant member,
                // NX for linear members,
                // 1 for temporary storage
                //
                fmatrix = new double[nq, 1+nx+1];
            }
            if( d==2 )
            {
                y = new double[nq];
                w = new double[nq];
                qsol = new double[1+nx+(int)Math.Round(nx*(nx+1)*0.5)];
                
                //
                // 1 for constant member,
                // NX for linear members,
                // Round(NX*(NX+1)*0.5) for quadratic model,
                // 1 for temporary storage
                //
                fmatrix = new double[nq, 1+nx+(int)Math.Round(nx*(nx+1)*0.5)+1];
            }
            for(i=0; i<=n-1; i++)
            {
                
                //
                // Initialize center.
                //
                for(i_=0; i_<=nx-1;i_++)
                {
                    z.q[i,i_] = xy[i,i_];
                }
                
                //
                // Calculate linear/quadratic members
                // using least squares fit
                // NOTE 1: all weight are equal to 1.0
                // NOTE 2: self-match is USED for this query
                //
                for(i_=0; i_<=nx-1;i_++)
                {
                    x[i_] = xy[i,i_];
                }
                k = nearestneighbor.kdtreequeryknn(z.tree, x, nq, true);
                nearestneighbor.kdtreequeryresultsxy(z.tree, ref qxybuf);
                nearestneighbor.kdtreequeryresultsdistances(z.tree, ref qrbuf);
                if( d==1 )
                {
                    
                    //
                    // Linear nodal function calculated using
                    // least squares fitting to its neighbors
                    //
                    for(j=0; j<=k-1; j++)
                    {
                        fmatrix[j,0] = 1.0;
                        for(j2=0; j2<=nx-1; j2++)
                        {
                            fmatrix[j,1+j2] = qxybuf[j,j2]-xy[i,j2];
                        }
                        y[j] = qxybuf[j,nx];
                        w[j] = 1;
                    }
                    nc = 1+nx;
                }
                if( d==2 )
                {
                    
                    //
                    // Quadratic nodal function calculated using
                    // least squares fitting to its neighbors
                    //
                    for(j=0; j<=k-1; j++)
                    {
                        fmatrix[j,0] = 1;
                        offs = 1;
                        for(j2=0; j2<=nx-1; j2++)
                        {
                            fmatrix[j,offs] = qxybuf[j,j2]-xy[i,j2];
                            offs = offs+1;
                        }
                        for(j2=0; j2<=nx-1; j2++)
                        {
                            for(j3=j2; j3<=nx-1; j3++)
                            {
                                fmatrix[j,offs] = (qxybuf[j,j2]-xy[i,j2])*(qxybuf[j,j3]-xy[i,j3]);
                                offs = offs+1;
                            }
                        }
                        y[j] = qxybuf[j,nx];
                        w[j] = 1;
                    }
                    nc = 1+nx+(int)Math.Round(nx*(nx+1)*0.5);
                }
                idwinternalsolver(ref y, ref w, ref fmatrix, ref temp, k, nc, ref info, ref qsol, ref taskrcond);
                
                //
                // Least squares models: copy results
                //
                if( info>0 )
                {
                    
                    //
                    // LLS task is solved, copy results
                    //
                    z.debugworstrcond = Math.Min(z.debugworstrcond, taskrcond);
                    z.debugbestrcond = Math.Max(z.debugbestrcond, taskrcond);
                    for(j=0; j<=nc-1; j++)
                    {
                        z.q[i,nx+j] = qsol[j];
                    }
                }
                else
                {
                    
                    //
                    // Solver failure, very strange, but we will use
                    // zero values to handle it.
                    //
                    z.debugsolverfailures = z.debugsolverfailures+1;
                    v = 0;
                    for(j=0; j<=k-1; j++)
                    {
                        v = v+qxybuf[j,nx];
                    }
                    z.q[i,nx] = v/k;
                    for(j=0; j<=nc-2; j++)
                    {
                        z.q[i,nx+1+j] = 0;
                    }
                }
            }
        }


        /*************************************************************************
        Internal subroutine: K-th nodal function calculation

          -- ALGLIB --
             Copyright 02.03.2010 by Bochkanov Sergey
        *************************************************************************/
        private static double idwcalcq(idwinterpolant z,
            double[] x,
            int k)
        {
            double result = 0;
            int nx = 0;
            int i = 0;
            int j = 0;
            int offs = 0;

            nx = z.nx;
            
            //
            // constant member
            //
            result = z.q[k,nx];
            
            //
            // linear members
            //
            if( z.d>=1 )
            {
                for(i=0; i<=nx-1; i++)
                {
                    result = result+z.q[k,nx+1+i]*(x[i]-z.q[k,i]);
                }
            }
            
            //
            // quadratic members
            //
            if( z.d>=2 )
            {
                offs = nx+1+nx;
                for(i=0; i<=nx-1; i++)
                {
                    for(j=i; j<=nx-1; j++)
                    {
                        result = result+z.q[k,offs]*(x[i]-z.q[k,i])*(x[j]-z.q[k,j]);
                        offs = offs+1;
                    }
                }
            }
            return result;
        }


        /*************************************************************************
        Initialization of internal structures.

        It assumes correctness of all parameters.

          -- ALGLIB --
             Copyright 02.03.2010 by Bochkanov Sergey
        *************************************************************************/
        private static void idwinit1(int n,
            int nx,
            int d,
            int nq,
            int nw,
            idwinterpolant z)
        {
            z.debugsolverfailures = 0;
            z.debugworstrcond = 1.0;
            z.debugbestrcond = 0;
            z.n = n;
            z.nx = nx;
            z.d = 0;
            if( d==1 )
            {
                z.d = 1;
            }
            if( d==2 )
            {
                z.d = 2;
            }
            if( d==-1 )
            {
                z.d = 1;
            }
            z.nw = nw;
            if( d==-1 )
            {
                z.q = new double[n, nx+1+nx];
            }
            if( d==0 )
            {
                z.q = new double[n, nx+1];
            }
            if( d==1 )
            {
                z.q = new double[n, nx+1+nx];
            }
            if( d==2 )
            {
                z.q = new double[n, nx+1+nx+(int)Math.Round(nx*(nx+1)*0.5)];
            }
            z.tbuf = new int[nw];
            z.rbuf = new double[nw];
            z.xybuf = new double[nw, nx+1];
            z.xbuf = new double[nx];
        }


        /*************************************************************************
        Linear least squares solver for small tasks.

        Works faster than standard ALGLIB solver in non-degenerate cases  (due  to
        absense of internal allocations and optimized row/colums).  In  degenerate
        cases it calls standard solver, which results in small performance penalty
        associated with preliminary steps.

        INPUT PARAMETERS:
            Y           array[0..N-1]
            W           array[0..N-1]
            FMatrix     array[0..N-1,0..M], have additional column for temporary
                        values
            Temp        array[0..N]
        *************************************************************************/
        private static void idwinternalsolver(ref double[] y,
            ref double[] w,
            ref double[,] fmatrix,
            ref double[] temp,
            int n,
            int m,
            ref int info,
            ref double[] x,
            ref double taskrcond)
        {
            int i = 0;
            int j = 0;
            double v = 0;
            double tau = 0;
            double[] b = new double[0];
            densesolver.densesolverlsreport srep = new densesolver.densesolverlsreport();
            int i_ = 0;
            int i1_ = 0;

            info = 0;

            
            //
            // set up info
            //
            info = 1;
            
            //
            // prepare matrix
            //
            for(i=0; i<=n-1; i++)
            {
                fmatrix[i,m] = y[i];
                v = w[i];
                for(i_=0; i_<=m;i_++)
                {
                    fmatrix[i,i_] = v*fmatrix[i,i_];
                }
            }
            
            //
            // use either fast algorithm or general algorithm
            //
            if( m<=n )
            {
                
                //
                // QR decomposition
                // We assume that M<=N (we would have called LSFit() otherwise)
                //
                for(i=0; i<=m-1; i++)
                {
                    if( i<n-1 )
                    {
                        i1_ = (i) - (1);
                        for(i_=1; i_<=n-i;i_++)
                        {
                            temp[i_] = fmatrix[i_+i1_,i];
                        }
                        reflections.generatereflection(ref temp, n-i, ref tau);
                        fmatrix[i,i] = temp[1];
                        temp[1] = 1;
                        for(j=i+1; j<=m; j++)
                        {
                            i1_ = (1)-(i);
                            v = 0.0;
                            for(i_=i; i_<=n-1;i_++)
                            {
                                v += fmatrix[i_,j]*temp[i_+i1_];
                            }
                            v = tau*v;
                            i1_ = (1) - (i);
                            for(i_=i; i_<=n-1;i_++)
                            {
                                fmatrix[i_,j] = fmatrix[i_,j] - v*temp[i_+i1_];
                            }
                        }
                    }
                }
                
                //
                // Check condition number
                //
                taskrcond = rcond.rmatrixtrrcondinf(fmatrix, m, true, false);
                
                //
                // use either fast algorithm for non-degenerate cases
                // or slow algorithm for degenerate cases
                //
                if( (double)(taskrcond)>(double)(10000*n*math.machineepsilon) )
                {
                    
                    //
                    // solve triangular system R*x = FMatrix[0:M-1,M]
                    // using fast algorithm, then exit
                    //
                    x[m-1] = fmatrix[m-1,m]/fmatrix[m-1,m-1];
                    for(i=m-2; i>=0; i--)
                    {
                        v = 0.0;
                        for(i_=i+1; i_<=m-1;i_++)
                        {
                            v += fmatrix[i,i_]*x[i_];
                        }
                        x[i] = (fmatrix[i,m]-v)/fmatrix[i,i];
                    }
                }
                else
                {
                    
                    //
                    // use more general algorithm
                    //
                    b = new double[m];
                    for(i=0; i<=m-1; i++)
                    {
                        for(j=0; j<=i-1; j++)
                        {
                            fmatrix[i,j] = 0.0;
                        }
                        b[i] = fmatrix[i,m];
                    }
                    densesolver.rmatrixsolvels(fmatrix, m, m, b, 10000*math.machineepsilon, ref info, srep, ref x);
                }
            }
            else
            {
                
                //
                // use more general algorithm
                //
                b = new double[n];
                for(i=0; i<=n-1; i++)
                {
                    b[i] = fmatrix[i,m];
                }
                densesolver.rmatrixsolvels(fmatrix, n, m, b, 10000*math.machineepsilon, ref info, srep, ref x);
                taskrcond = srep.r2;
            }
        }


    }
    public class ratint
    {
        /*************************************************************************
        Barycentric interpolant.
        *************************************************************************/
        public class barycentricinterpolant
        {
            public int n;
            public double sy;
            public double[] x;
            public double[] y;
            public double[] w;
            public barycentricinterpolant()
            {
                x = new double[0];
                y = new double[0];
                w = new double[0];
            }
        };




        /*************************************************************************
        Rational interpolation using barycentric formula

        F(t) = SUM(i=0,n-1,w[i]*f[i]/(t-x[i])) / SUM(i=0,n-1,w[i]/(t-x[i]))

        Input parameters:
            B   -   barycentric interpolant built with one of model building
                    subroutines.
            T   -   interpolation point

        Result:
            barycentric interpolant F(t)

          -- ALGLIB --
             Copyright 17.08.2009 by Bochkanov Sergey
        *************************************************************************/
        public static double barycentriccalc(barycentricinterpolant b,
            double t)
        {
            double result = 0;
            double s1 = 0;
            double s2 = 0;
            double s = 0;
            double v = 0;
            int i = 0;

            alglib.ap.assert(!Double.IsInfinity(t), "BarycentricCalc: infinite T!");
            
            //
            // special case: NaN
            //
            if( Double.IsNaN(t) )
            {
                result = Double.NaN;
                return result;
            }
            
            //
            // special case: N=1
            //
            if( b.n==1 )
            {
                result = b.sy*b.y[0];
                return result;
            }
            
            //
            // Here we assume that task is normalized, i.e.:
            // 1. abs(Y[i])<=1
            // 2. abs(W[i])<=1
            // 3. X[] is ordered
            //
            s = Math.Abs(t-b.x[0]);
            for(i=0; i<=b.n-1; i++)
            {
                v = b.x[i];
                if( (double)(v)==(double)(t) )
                {
                    result = b.sy*b.y[i];
                    return result;
                }
                v = Math.Abs(t-v);
                if( (double)(v)<(double)(s) )
                {
                    s = v;
                }
            }
            s1 = 0;
            s2 = 0;
            for(i=0; i<=b.n-1; i++)
            {
                v = s/(t-b.x[i]);
                v = v*b.w[i];
                s1 = s1+v*b.y[i];
                s2 = s2+v;
            }
            result = b.sy*s1/s2;
            return result;
        }


        /*************************************************************************
        Differentiation of barycentric interpolant: first derivative.

        Algorithm used in this subroutine is very robust and should not fail until
        provided with values too close to MaxRealNumber  (usually  MaxRealNumber/N
        or greater will overflow).

        INPUT PARAMETERS:
            B   -   barycentric interpolant built with one of model building
                    subroutines.
            T   -   interpolation point

        OUTPUT PARAMETERS:
            F   -   barycentric interpolant at T
            DF  -   first derivative
            
        NOTE


          -- ALGLIB --
             Copyright 17.08.2009 by Bochkanov Sergey
        *************************************************************************/
        public static void barycentricdiff1(barycentricinterpolant b,
            double t,
            ref double f,
            ref double df)
        {
            double v = 0;
            double vv = 0;
            int i = 0;
            int k = 0;
            double n0 = 0;
            double n1 = 0;
            double d0 = 0;
            double d1 = 0;
            double s0 = 0;
            double s1 = 0;
            double xk = 0;
            double xi = 0;
            double xmin = 0;
            double xmax = 0;
            double xscale1 = 0;
            double xoffs1 = 0;
            double xscale2 = 0;
            double xoffs2 = 0;
            double xprev = 0;

            f = 0;
            df = 0;

            alglib.ap.assert(!Double.IsInfinity(t), "BarycentricDiff1: infinite T!");
            
            //
            // special case: NaN
            //
            if( Double.IsNaN(t) )
            {
                f = Double.NaN;
                df = Double.NaN;
                return;
            }
            
            //
            // special case: N=1
            //
            if( b.n==1 )
            {
                f = b.sy*b.y[0];
                df = 0;
                return;
            }
            if( (double)(b.sy)==(double)(0) )
            {
                f = 0;
                df = 0;
                return;
            }
            alglib.ap.assert((double)(b.sy)>(double)(0), "BarycentricDiff1: internal error");
            
            //
            // We assume than N>1 and B.SY>0. Find:
            // 1. pivot point (X[i] closest to T)
            // 2. width of interval containing X[i]
            //
            v = Math.Abs(b.x[0]-t);
            k = 0;
            xmin = b.x[0];
            xmax = b.x[0];
            for(i=1; i<=b.n-1; i++)
            {
                vv = b.x[i];
                if( (double)(Math.Abs(vv-t))<(double)(v) )
                {
                    v = Math.Abs(vv-t);
                    k = i;
                }
                xmin = Math.Min(xmin, vv);
                xmax = Math.Max(xmax, vv);
            }
            
            //
            // pivot point found, calculate dNumerator and dDenominator
            //
            xscale1 = 1/(xmax-xmin);
            xoffs1 = -(xmin/(xmax-xmin))+1;
            xscale2 = 2;
            xoffs2 = -3;
            t = t*xscale1+xoffs1;
            t = t*xscale2+xoffs2;
            xk = b.x[k];
            xk = xk*xscale1+xoffs1;
            xk = xk*xscale2+xoffs2;
            v = t-xk;
            n0 = 0;
            n1 = 0;
            d0 = 0;
            d1 = 0;
            xprev = -2;
            for(i=0; i<=b.n-1; i++)
            {
                xi = b.x[i];
                xi = xi*xscale1+xoffs1;
                xi = xi*xscale2+xoffs2;
                alglib.ap.assert((double)(xi)>(double)(xprev), "BarycentricDiff1: points are too close!");
                xprev = xi;
                if( i!=k )
                {
                    vv = math.sqr(t-xi);
                    s0 = (t-xk)/(t-xi);
                    s1 = (xk-xi)/vv;
                }
                else
                {
                    s0 = 1;
                    s1 = 0;
                }
                vv = b.w[i]*b.y[i];
                n0 = n0+s0*vv;
                n1 = n1+s1*vv;
                vv = b.w[i];
                d0 = d0+s0*vv;
                d1 = d1+s1*vv;
            }
            f = b.sy*n0/d0;
            df = (n1*d0-n0*d1)/math.sqr(d0);
            if( (double)(df)!=(double)(0) )
            {
                df = Math.Sign(df)*Math.Exp(Math.Log(Math.Abs(df))+Math.Log(b.sy)+Math.Log(xscale1)+Math.Log(xscale2));
            }
        }


        /*************************************************************************
        Differentiation of barycentric interpolant: first/second derivatives.

        INPUT PARAMETERS:
            B   -   barycentric interpolant built with one of model building
                    subroutines.
            T   -   interpolation point

        OUTPUT PARAMETERS:
            F   -   barycentric interpolant at T
            DF  -   first derivative
            D2F -   second derivative

        NOTE: this algorithm may fail due to overflow/underflor if  used  on  data
        whose values are close to MaxRealNumber or MinRealNumber.  Use more robust
        BarycentricDiff1() subroutine in such cases.


          -- ALGLIB --
             Copyright 17.08.2009 by Bochkanov Sergey
        *************************************************************************/
        public static void barycentricdiff2(barycentricinterpolant b,
            double t,
            ref double f,
            ref double df,
            ref double d2f)
        {
            double v = 0;
            double vv = 0;
            int i = 0;
            int k = 0;
            double n0 = 0;
            double n1 = 0;
            double n2 = 0;
            double d0 = 0;
            double d1 = 0;
            double d2 = 0;
            double s0 = 0;
            double s1 = 0;
            double s2 = 0;
            double xk = 0;
            double xi = 0;

            f = 0;
            df = 0;
            d2f = 0;

            alglib.ap.assert(!Double.IsInfinity(t), "BarycentricDiff1: infinite T!");
            
            //
            // special case: NaN
            //
            if( Double.IsNaN(t) )
            {
                f = Double.NaN;
                df = Double.NaN;
                d2f = Double.NaN;
                return;
            }
            
            //
            // special case: N=1
            //
            if( b.n==1 )
            {
                f = b.sy*b.y[0];
                df = 0;
                d2f = 0;
                return;
            }
            if( (double)(b.sy)==(double)(0) )
            {
                f = 0;
                df = 0;
                d2f = 0;
                return;
            }
            
            //
            // We assume than N>1 and B.SY>0. Find:
            // 1. pivot point (X[i] closest to T)
            // 2. width of interval containing X[i]
            //
            alglib.ap.assert((double)(b.sy)>(double)(0), "BarycentricDiff: internal error");
            f = 0;
            df = 0;
            d2f = 0;
            v = Math.Abs(b.x[0]-t);
            k = 0;
            for(i=1; i<=b.n-1; i++)
            {
                vv = b.x[i];
                if( (double)(Math.Abs(vv-t))<(double)(v) )
                {
                    v = Math.Abs(vv-t);
                    k = i;
                }
            }
            
            //
            // pivot point found, calculate dNumerator and dDenominator
            //
            xk = b.x[k];
            v = t-xk;
            n0 = 0;
            n1 = 0;
            n2 = 0;
            d0 = 0;
            d1 = 0;
            d2 = 0;
            for(i=0; i<=b.n-1; i++)
            {
                if( i!=k )
                {
                    xi = b.x[i];
                    vv = math.sqr(t-xi);
                    s0 = (t-xk)/(t-xi);
                    s1 = (xk-xi)/vv;
                    s2 = -(2*(xk-xi)/(vv*(t-xi)));
                }
                else
                {
                    s0 = 1;
                    s1 = 0;
                    s2 = 0;
                }
                vv = b.w[i]*b.y[i];
                n0 = n0+s0*vv;
                n1 = n1+s1*vv;
                n2 = n2+s2*vv;
                vv = b.w[i];
                d0 = d0+s0*vv;
                d1 = d1+s1*vv;
                d2 = d2+s2*vv;
            }
            f = b.sy*n0/d0;
            df = b.sy*(n1*d0-n0*d1)/math.sqr(d0);
            d2f = b.sy*((n2*d0-n0*d2)*math.sqr(d0)-(n1*d0-n0*d1)*2*d0*d1)/math.sqr(math.sqr(d0));
        }


        /*************************************************************************
        This subroutine performs linear transformation of the argument.

        INPUT PARAMETERS:
            B       -   rational interpolant in barycentric form
            CA, CB  -   transformation coefficients: x = CA*t + CB

        OUTPUT PARAMETERS:
            B       -   transformed interpolant with X replaced by T

          -- ALGLIB PROJECT --
             Copyright 19.08.2009 by Bochkanov Sergey
        *************************************************************************/
        public static void barycentriclintransx(barycentricinterpolant b,
            double ca,
            double cb)
        {
            int i = 0;
            int j = 0;
            double v = 0;

            
            //
            // special case, replace by constant F(CB)
            //
            if( (double)(ca)==(double)(0) )
            {
                b.sy = barycentriccalc(b, cb);
                v = 1;
                for(i=0; i<=b.n-1; i++)
                {
                    b.y[i] = 1;
                    b.w[i] = v;
                    v = -v;
                }
                return;
            }
            
            //
            // general case: CA<>0
            //
            for(i=0; i<=b.n-1; i++)
            {
                b.x[i] = (b.x[i]-cb)/ca;
            }
            if( (double)(ca)<(double)(0) )
            {
                for(i=0; i<=b.n-1; i++)
                {
                    if( i<b.n-1-i )
                    {
                        j = b.n-1-i;
                        v = b.x[i];
                        b.x[i] = b.x[j];
                        b.x[j] = v;
                        v = b.y[i];
                        b.y[i] = b.y[j];
                        b.y[j] = v;
                        v = b.w[i];
                        b.w[i] = b.w[j];
                        b.w[j] = v;
                    }
                    else
                    {
                        break;
                    }
                }
            }
        }


        /*************************************************************************
        This  subroutine   performs   linear  transformation  of  the  barycentric
        interpolant.

        INPUT PARAMETERS:
            B       -   rational interpolant in barycentric form
            CA, CB  -   transformation coefficients: B2(x) = CA*B(x) + CB

        OUTPUT PARAMETERS:
            B       -   transformed interpolant

          -- ALGLIB PROJECT --
             Copyright 19.08.2009 by Bochkanov Sergey
        *************************************************************************/
        public static void barycentriclintransy(barycentricinterpolant b,
            double ca,
            double cb)
        {
            int i = 0;
            double v = 0;
            int i_ = 0;

            for(i=0; i<=b.n-1; i++)
            {
                b.y[i] = ca*b.sy*b.y[i]+cb;
            }
            b.sy = 0;
            for(i=0; i<=b.n-1; i++)
            {
                b.sy = Math.Max(b.sy, Math.Abs(b.y[i]));
            }
            if( (double)(b.sy)>(double)(0) )
            {
                v = 1/b.sy;
                for(i_=0; i_<=b.n-1;i_++)
                {
                    b.y[i_] = v*b.y[i_];
                }
            }
        }


        /*************************************************************************
        Extracts X/Y/W arrays from rational interpolant

        INPUT PARAMETERS:
            B   -   barycentric interpolant

        OUTPUT PARAMETERS:
            N   -   nodes count, N>0
            X   -   interpolation nodes, array[0..N-1]
            F   -   function values, array[0..N-1]
            W   -   barycentric weights, array[0..N-1]

          -- ALGLIB --
             Copyright 17.08.2009 by Bochkanov Sergey
        *************************************************************************/
        public static void barycentricunpack(barycentricinterpolant b,
            ref int n,
            ref double[] x,
            ref double[] y,
            ref double[] w)
        {
            double v = 0;
            int i_ = 0;

            n = 0;
            x = new double[0];
            y = new double[0];
            w = new double[0];

            n = b.n;
            x = new double[n];
            y = new double[n];
            w = new double[n];
            v = b.sy;
            for(i_=0; i_<=n-1;i_++)
            {
                x[i_] = b.x[i_];
            }
            for(i_=0; i_<=n-1;i_++)
            {
                y[i_] = v*b.y[i_];
            }
            for(i_=0; i_<=n-1;i_++)
            {
                w[i_] = b.w[i_];
            }
        }


        /*************************************************************************
        Rational interpolant from X/Y/W arrays

        F(t) = SUM(i=0,n-1,w[i]*f[i]/(t-x[i])) / SUM(i=0,n-1,w[i]/(t-x[i]))

        INPUT PARAMETERS:
            X   -   interpolation nodes, array[0..N-1]
            F   -   function values, array[0..N-1]
            W   -   barycentric weights, array[0..N-1]
            N   -   nodes count, N>0

        OUTPUT PARAMETERS:
            B   -   barycentric interpolant built from (X, Y, W)

          -- ALGLIB --
             Copyright 17.08.2009 by Bochkanov Sergey
        *************************************************************************/
        public static void barycentricbuildxyw(double[] x,
            double[] y,
            double[] w,
            int n,
            barycentricinterpolant b)
        {
            int i_ = 0;

            alglib.ap.assert(n>0, "BarycentricBuildXYW: incorrect N!");
            
            //
            // fill X/Y/W
            //
            b.x = new double[n];
            b.y = new double[n];
            b.w = new double[n];
            for(i_=0; i_<=n-1;i_++)
            {
                b.x[i_] = x[i_];
            }
            for(i_=0; i_<=n-1;i_++)
            {
                b.y[i_] = y[i_];
            }
            for(i_=0; i_<=n-1;i_++)
            {
                b.w[i_] = w[i_];
            }
            b.n = n;
            
            //
            // Normalize
            //
            barycentricnormalize(b);
        }


        /*************************************************************************
        Rational interpolant without poles

        The subroutine constructs the rational interpolating function without real
        poles  (see  'Barycentric rational interpolation with no  poles  and  high
        rates of approximation', Michael S. Floater. and  Kai  Hormann,  for  more
        information on this subject).

        Input parameters:
            X   -   interpolation nodes, array[0..N-1].
            Y   -   function values, array[0..N-1].
            N   -   number of nodes, N>0.
            D   -   order of the interpolation scheme, 0 <= D <= N-1.
                    D<0 will cause an error.
                    D>=N it will be replaced with D=N-1.
                    if you don't know what D to choose, use small value about 3-5.

        Output parameters:
            B   -   barycentric interpolant.

        Note:
            this algorithm always succeeds and calculates the weights  with  close
            to machine precision.

          -- ALGLIB PROJECT --
             Copyright 17.06.2007 by Bochkanov Sergey
        *************************************************************************/
        public static void barycentricbuildfloaterhormann(double[] x,
            double[] y,
            int n,
            int d,
            barycentricinterpolant b)
        {
            double s0 = 0;
            double s = 0;
            double v = 0;
            int i = 0;
            int j = 0;
            int k = 0;
            int[] perm = new int[0];
            double[] wtemp = new double[0];
            double[] sortrbuf = new double[0];
            double[] sortrbuf2 = new double[0];
            int i_ = 0;

            alglib.ap.assert(n>0, "BarycentricFloaterHormann: N<=0!");
            alglib.ap.assert(d>=0, "BarycentricFloaterHormann: incorrect D!");
            
            //
            // Prepare
            //
            if( d>n-1 )
            {
                d = n-1;
            }
            b.n = n;
            
            //
            // special case: N=1
            //
            if( n==1 )
            {
                b.x = new double[n];
                b.y = new double[n];
                b.w = new double[n];
                b.x[0] = x[0];
                b.y[0] = y[0];
                b.w[0] = 1;
                barycentricnormalize(b);
                return;
            }
            
            //
            // Fill X/Y
            //
            b.x = new double[n];
            b.y = new double[n];
            for(i_=0; i_<=n-1;i_++)
            {
                b.x[i_] = x[i_];
            }
            for(i_=0; i_<=n-1;i_++)
            {
                b.y[i_] = y[i_];
            }
            tsort.tagsortfastr(ref b.x, ref b.y, ref sortrbuf, ref sortrbuf2, n);
            
            //
            // Calculate Wk
            //
            b.w = new double[n];
            s0 = 1;
            for(k=1; k<=d; k++)
            {
                s0 = -s0;
            }
            for(k=0; k<=n-1; k++)
            {
                
                //
                // Wk
                //
                s = 0;
                for(i=Math.Max(k-d, 0); i<=Math.Min(k, n-1-d); i++)
                {
                    v = 1;
                    for(j=i; j<=i+d; j++)
                    {
                        if( j!=k )
                        {
                            v = v/Math.Abs(b.x[k]-b.x[j]);
                        }
                    }
                    s = s+v;
                }
                b.w[k] = s0*s;
                
                //
                // Next S0
                //
                s0 = -s0;
            }
            
            //
            // Normalize
            //
            barycentricnormalize(b);
        }


        /*************************************************************************
        Copying of the barycentric interpolant (for internal use only)

        INPUT PARAMETERS:
            B   -   barycentric interpolant

        OUTPUT PARAMETERS:
            B2  -   copy(B1)

          -- ALGLIB --
             Copyright 17.08.2009 by Bochkanov Sergey
        *************************************************************************/
        public static void barycentriccopy(barycentricinterpolant b,
            barycentricinterpolant b2)
        {
            int i_ = 0;

            b2.n = b.n;
            b2.sy = b.sy;
            b2.x = new double[b2.n];
            b2.y = new double[b2.n];
            b2.w = new double[b2.n];
            for(i_=0; i_<=b2.n-1;i_++)
            {
                b2.x[i_] = b.x[i_];
            }
            for(i_=0; i_<=b2.n-1;i_++)
            {
                b2.y[i_] = b.y[i_];
            }
            for(i_=0; i_<=b2.n-1;i_++)
            {
                b2.w[i_] = b.w[i_];
            }
        }


        /*************************************************************************
        Normalization of barycentric interpolant:
        * B.N, B.X, B.Y and B.W are initialized
        * B.SY is NOT initialized
        * Y[] is normalized, scaling coefficient is stored in B.SY
        * W[] is normalized, no scaling coefficient is stored
        * X[] is sorted

        Internal subroutine.
        *************************************************************************/
        private static void barycentricnormalize(barycentricinterpolant b)
        {
            int[] p1 = new int[0];
            int[] p2 = new int[0];
            int i = 0;
            int j = 0;
            int j2 = 0;
            double v = 0;
            int i_ = 0;

            
            //
            // Normalize task: |Y|<=1, |W|<=1, sort X[]
            //
            b.sy = 0;
            for(i=0; i<=b.n-1; i++)
            {
                b.sy = Math.Max(b.sy, Math.Abs(b.y[i]));
            }
            if( (double)(b.sy)>(double)(0) && (double)(Math.Abs(b.sy-1))>(double)(10*math.machineepsilon) )
            {
                v = 1/b.sy;
                for(i_=0; i_<=b.n-1;i_++)
                {
                    b.y[i_] = v*b.y[i_];
                }
            }
            v = 0;
            for(i=0; i<=b.n-1; i++)
            {
                v = Math.Max(v, Math.Abs(b.w[i]));
            }
            if( (double)(v)>(double)(0) && (double)(Math.Abs(v-1))>(double)(10*math.machineepsilon) )
            {
                v = 1/v;
                for(i_=0; i_<=b.n-1;i_++)
                {
                    b.w[i_] = v*b.w[i_];
                }
            }
            for(i=0; i<=b.n-2; i++)
            {
                if( (double)(b.x[i+1])<(double)(b.x[i]) )
                {
                    tsort.tagsort(ref b.x, b.n, ref p1, ref p2);
                    for(j=0; j<=b.n-1; j++)
                    {
                        j2 = p2[j];
                        v = b.y[j];
                        b.y[j] = b.y[j2];
                        b.y[j2] = v;
                        v = b.w[j];
                        b.w[j] = b.w[j2];
                        b.w[j2] = v;
                    }
                    break;
                }
            }
        }


    }
    public class polint
    {
        /*************************************************************************
        Conversion from barycentric representation to Chebyshev basis.
        This function has O(N^2) complexity.

        INPUT PARAMETERS:
            P   -   polynomial in barycentric form
            A,B -   base interval for Chebyshev polynomials (see below)
                    A<>B

        OUTPUT PARAMETERS
            T   -   coefficients of Chebyshev representation;
                    P(x) = sum { T[i]*Ti(2*(x-A)/(B-A)-1), i=0..N-1 },
                    where Ti - I-th Chebyshev polynomial.

        NOTES:
            barycentric interpolant passed as P may be either polynomial  obtained
            from  polynomial  interpolation/ fitting or rational function which is
            NOT polynomial. We can't distinguish between these two cases, and this
            algorithm just tries to work assuming that P IS a polynomial.  If not,
            algorithm will return results, but they won't have any meaning.

          -- ALGLIB --
             Copyright 30.09.2010 by Bochkanov Sergey
        *************************************************************************/
        public static void polynomialbar2cheb(ratint.barycentricinterpolant p,
            double a,
            double b,
            ref double[] t)
        {
            int i = 0;
            int k = 0;
            double[] vp = new double[0];
            double[] vx = new double[0];
            double[] tk = new double[0];
            double[] tk1 = new double[0];
            double v = 0;
            int i_ = 0;

            t = new double[0];

            alglib.ap.assert(math.isfinite(a), "PolynomialBar2Cheb: A is not finite!");
            alglib.ap.assert(math.isfinite(b), "PolynomialBar2Cheb: B is not finite!");
            alglib.ap.assert((double)(a)!=(double)(b), "PolynomialBar2Cheb: A=B!");
            alglib.ap.assert(p.n>0, "PolynomialBar2Cheb: P is not correctly initialized barycentric interpolant!");
            
            //
            // Calculate function values on a Chebyshev grid
            //
            vp = new double[p.n];
            vx = new double[p.n];
            for(i=0; i<=p.n-1; i++)
            {
                vx[i] = Math.Cos(Math.PI*(i+0.5)/p.n);
                vp[i] = ratint.barycentriccalc(p, 0.5*(vx[i]+1)*(b-a)+a);
            }
            
            //
            // T[0]
            //
            t = new double[p.n];
            v = 0;
            for(i=0; i<=p.n-1; i++)
            {
                v = v+vp[i];
            }
            t[0] = v/p.n;
            
            //
            // other T's.
            //
            // NOTES:
            // 1. TK stores T{k} on VX, TK1 stores T{k-1} on VX
            // 2. we can do same calculations with fast DCT, but it
            //    * adds dependencies
            //    * still leaves us with O(N^2) algorithm because
            //      preparation of function values is O(N^2) process
            //
            if( p.n>1 )
            {
                tk = new double[p.n];
                tk1 = new double[p.n];
                for(i=0; i<=p.n-1; i++)
                {
                    tk[i] = vx[i];
                    tk1[i] = 1;
                }
                for(k=1; k<=p.n-1; k++)
                {
                    
                    //
                    // calculate discrete product of function vector and TK
                    //
                    v = 0.0;
                    for(i_=0; i_<=p.n-1;i_++)
                    {
                        v += tk[i_]*vp[i_];
                    }
                    t[k] = v/(0.5*p.n);
                    
                    //
                    // Update TK and TK1
                    //
                    for(i=0; i<=p.n-1; i++)
                    {
                        v = 2*vx[i]*tk[i]-tk1[i];
                        tk1[i] = tk[i];
                        tk[i] = v;
                    }
                }
            }
        }


        /*************************************************************************
        Conversion from Chebyshev basis to barycentric representation.
        This function has O(N^2) complexity.

        INPUT PARAMETERS:
            T   -   coefficients of Chebyshev representation;
                    P(x) = sum { T[i]*Ti(2*(x-A)/(B-A)-1), i=0..N },
                    where Ti - I-th Chebyshev polynomial.
            N   -   number of coefficients:
                    * if given, only leading N elements of T are used
                    * if not given, automatically determined from size of T
            A,B -   base interval for Chebyshev polynomials (see above)
                    A<B

        OUTPUT PARAMETERS
            P   -   polynomial in barycentric form

          -- ALGLIB --
             Copyright 30.09.2010 by Bochkanov Sergey
        *************************************************************************/
        public static void polynomialcheb2bar(double[] t,
            int n,
            double a,
            double b,
            ratint.barycentricinterpolant p)
        {
            int i = 0;
            int k = 0;
            double[] y = new double[0];
            double tk = 0;
            double tk1 = 0;
            double vx = 0;
            double vy = 0;
            double v = 0;

            alglib.ap.assert(math.isfinite(a), "PolynomialBar2Cheb: A is not finite!");
            alglib.ap.assert(math.isfinite(b), "PolynomialBar2Cheb: B is not finite!");
            alglib.ap.assert((double)(a)!=(double)(b), "PolynomialBar2Cheb: A=B!");
            alglib.ap.assert(n>=1, "PolynomialBar2Cheb: N<1");
            alglib.ap.assert(alglib.ap.len(t)>=n, "PolynomialBar2Cheb: Length(T)<N");
            alglib.ap.assert(apserv.isfinitevector(t, n), "PolynomialBar2Cheb: T[] contains INF or NAN");
            
            //
            // Calculate function values on a Chebyshev grid spanning [-1,+1]
            //
            y = new double[n];
            for(i=0; i<=n-1; i++)
            {
                
                //
                // Calculate value on a grid spanning [-1,+1]
                //
                vx = Math.Cos(Math.PI*(i+0.5)/n);
                vy = t[0];
                tk1 = 1;
                tk = vx;
                for(k=1; k<=n-1; k++)
                {
                    vy = vy+t[k]*tk;
                    v = 2*vx*tk-tk1;
                    tk1 = tk;
                    tk = v;
                }
                y[i] = vy;
            }
            
            //
            // Build barycentric interpolant, map grid from [-1,+1] to [A,B]
            //
            polynomialbuildcheb1(a, b, y, n, p);
        }


        /*************************************************************************
        Conversion from barycentric representation to power basis.
        This function has O(N^2) complexity.

        INPUT PARAMETERS:
            P   -   polynomial in barycentric form
            C   -   offset (see below); 0.0 is used as default value.
            S   -   scale (see below);  1.0 is used as default value. S<>0.

        OUTPUT PARAMETERS
            A   -   coefficients, P(x) = sum { A[i]*((X-C)/S)^i, i=0..N-1 }
            N   -   number of coefficients (polynomial degree plus 1)

        NOTES:
        1.  this function accepts offset and scale, which can be  set  to  improve
            numerical properties of polynomial. For example, if P was obtained  as
            result of interpolation on [-1,+1],  you  can  set  C=0  and  S=1  and
            represent  P  as sum of 1, x, x^2, x^3 and so on. In most cases you it
            is exactly what you need.

            However, if your interpolation model was built on [999,1001], you will
            see significant growth of numerical errors when using {1, x, x^2, x^3}
            as basis. Representing P as sum of 1, (x-1000), (x-1000)^2, (x-1000)^3
            will be better option. Such representation can be  obtained  by  using
            1000.0 as offset C and 1.0 as scale S.

        2.  power basis is ill-conditioned and tricks described above can't  solve
            this problem completely. This function  will  return  coefficients  in
            any  case,  but  for  N>8  they  will  become unreliable. However, N's
            less than 5 are pretty safe.
            
        3.  barycentric interpolant passed as P may be either polynomial  obtained
            from  polynomial  interpolation/ fitting or rational function which is
            NOT polynomial. We can't distinguish between these two cases, and this
            algorithm just tries to work assuming that P IS a polynomial.  If not,
            algorithm will return results, but they won't have any meaning.

          -- ALGLIB --
             Copyright 30.09.2010 by Bochkanov Sergey
        *************************************************************************/
        public static void polynomialbar2pow(ratint.barycentricinterpolant p,
            double c,
            double s,
            ref double[] a)
        {
            int i = 0;
            int k = 0;
            double e = 0;
            double d = 0;
            double[] vp = new double[0];
            double[] vx = new double[0];
            double[] tk = new double[0];
            double[] tk1 = new double[0];
            double[] t = new double[0];
            double v = 0;
            int i_ = 0;

            a = new double[0];

            alglib.ap.assert(math.isfinite(c), "PolynomialBar2Pow: C is not finite!");
            alglib.ap.assert(math.isfinite(s), "PolynomialBar2Pow: S is not finite!");
            alglib.ap.assert((double)(s)!=(double)(0), "PolynomialBar2Pow: S=0!");
            alglib.ap.assert(p.n>0, "PolynomialBar2Pow: P is not correctly initialized barycentric interpolant!");
            
            //
            // Calculate function values on a Chebyshev grid
            //
            vp = new double[p.n];
            vx = new double[p.n];
            for(i=0; i<=p.n-1; i++)
            {
                vx[i] = Math.Cos(Math.PI*(i+0.5)/p.n);
                vp[i] = ratint.barycentriccalc(p, s*vx[i]+c);
            }
            
            //
            // T[0]
            //
            t = new double[p.n];
            v = 0;
            for(i=0; i<=p.n-1; i++)
            {
                v = v+vp[i];
            }
            t[0] = v/p.n;
            
            //
            // other T's.
            //
            // NOTES:
            // 1. TK stores T{k} on VX, TK1 stores T{k-1} on VX
            // 2. we can do same calculations with fast DCT, but it
            //    * adds dependencies
            //    * still leaves us with O(N^2) algorithm because
            //      preparation of function values is O(N^2) process
            //
            if( p.n>1 )
            {
                tk = new double[p.n];
                tk1 = new double[p.n];
                for(i=0; i<=p.n-1; i++)
                {
                    tk[i] = vx[i];
                    tk1[i] = 1;
                }
                for(k=1; k<=p.n-1; k++)
                {
                    
                    //
                    // calculate discrete product of function vector and TK
                    //
                    v = 0.0;
                    for(i_=0; i_<=p.n-1;i_++)
                    {
                        v += tk[i_]*vp[i_];
                    }
                    t[k] = v/(0.5*p.n);
                    
                    //
                    // Update TK and TK1
                    //
                    for(i=0; i<=p.n-1; i++)
                    {
                        v = 2*vx[i]*tk[i]-tk1[i];
                        tk1[i] = tk[i];
                        tk[i] = v;
                    }
                }
            }
            
            //
            // Convert from Chebyshev basis to power basis
            //
            a = new double[p.n];
            for(i=0; i<=p.n-1; i++)
            {
                a[i] = 0;
            }
            d = 0;
            for(i=0; i<=p.n-1; i++)
            {
                for(k=i; k<=p.n-1; k++)
                {
                    e = a[k];
                    a[k] = 0;
                    if( i<=1 && k==i )
                    {
                        a[k] = 1;
                    }
                    else
                    {
                        if( i!=0 )
                        {
                            a[k] = 2*d;
                        }
                        if( k>i+1 )
                        {
                            a[k] = a[k]-a[k-2];
                        }
                    }
                    d = e;
                }
                d = a[i];
                e = 0;
                k = i;
                while( k<=p.n-1 )
                {
                    e = e+a[k]*t[k];
                    k = k+2;
                }
                a[i] = e;
            }
        }


        /*************************************************************************
        Conversion from power basis to barycentric representation.
        This function has O(N^2) complexity.

        INPUT PARAMETERS:
            A   -   coefficients, P(x) = sum { A[i]*((X-C)/S)^i, i=0..N-1 }
            N   -   number of coefficients (polynomial degree plus 1)
                    * if given, only leading N elements of A are used
                    * if not given, automatically determined from size of A
            C   -   offset (see below); 0.0 is used as default value.
            S   -   scale (see below);  1.0 is used as default value. S<>0.

        OUTPUT PARAMETERS
            P   -   polynomial in barycentric form


        NOTES:
        1.  this function accepts offset and scale, which can be  set  to  improve
            numerical properties of polynomial. For example, if you interpolate on
            [-1,+1],  you  can  set C=0 and S=1 and convert from sum of 1, x, x^2,
            x^3 and so on. In most cases you it is exactly what you need.

            However, if your interpolation model was built on [999,1001], you will
            see significant growth of numerical errors when using {1, x, x^2, x^3}
            as  input  basis.  Converting  from  sum  of  1, (x-1000), (x-1000)^2,
            (x-1000)^3 will be better option (you have to specify 1000.0 as offset
            C and 1.0 as scale S).

        2.  power basis is ill-conditioned and tricks described above can't  solve
            this problem completely. This function  will  return barycentric model
            in any case, but for N>8 accuracy well degrade. However, N's less than
            5 are pretty safe.

          -- ALGLIB --
             Copyright 30.09.2010 by Bochkanov Sergey
        *************************************************************************/
        public static void polynomialpow2bar(double[] a,
            int n,
            double c,
            double s,
            ratint.barycentricinterpolant p)
        {
            int i = 0;
            int k = 0;
            double[] y = new double[0];
            double vx = 0;
            double vy = 0;
            double px = 0;

            alglib.ap.assert(math.isfinite(c), "PolynomialPow2Bar: C is not finite!");
            alglib.ap.assert(math.isfinite(s), "PolynomialPow2Bar: S is not finite!");
            alglib.ap.assert((double)(s)!=(double)(0), "PolynomialPow2Bar: S is zero!");
            alglib.ap.assert(n>=1, "PolynomialPow2Bar: N<1");
            alglib.ap.assert(alglib.ap.len(a)>=n, "PolynomialPow2Bar: Length(A)<N");
            alglib.ap.assert(apserv.isfinitevector(a, n), "PolynomialPow2Bar: A[] contains INF or NAN");
            
            //
            // Calculate function values on a Chebyshev grid spanning [-1,+1]
            //
            y = new double[n];
            for(i=0; i<=n-1; i++)
            {
                
                //
                // Calculate value on a grid spanning [-1,+1]
                //
                vx = Math.Cos(Math.PI*(i+0.5)/n);
                vy = a[0];
                px = vx;
                for(k=1; k<=n-1; k++)
                {
                    vy = vy+px*a[k];
                    px = px*vx;
                }
                y[i] = vy;
            }
            
            //
            // Build barycentric interpolant, map grid from [-1,+1] to [A,B]
            //
            polynomialbuildcheb1(c-s, c+s, y, n, p);
        }


        /*************************************************************************
        Lagrange intepolant: generation of the model on the general grid.
        This function has O(N^2) complexity.

        INPUT PARAMETERS:
            X   -   abscissas, array[0..N-1]
            Y   -   function values, array[0..N-1]
            N   -   number of points, N>=1

        OUTPUT PARAMETERS
            P   -   barycentric model which represents Lagrange interpolant
                    (see ratint unit info and BarycentricCalc() description for
                    more information).

          -- ALGLIB --
             Copyright 02.12.2009 by Bochkanov Sergey
        *************************************************************************/
        public static void polynomialbuild(double[] x,
            double[] y,
            int n,
            ratint.barycentricinterpolant p)
        {
            int j = 0;
            int k = 0;
            double[] w = new double[0];
            double b = 0;
            double a = 0;
            double v = 0;
            double mx = 0;
            double[] sortrbuf = new double[0];
            double[] sortrbuf2 = new double[0];
            int i_ = 0;

            x = (double[])x.Clone();
            y = (double[])y.Clone();

            alglib.ap.assert(n>0, "PolynomialBuild: N<=0!");
            alglib.ap.assert(alglib.ap.len(x)>=n, "PolynomialBuild: Length(X)<N!");
            alglib.ap.assert(alglib.ap.len(y)>=n, "PolynomialBuild: Length(Y)<N!");
            alglib.ap.assert(apserv.isfinitevector(x, n), "PolynomialBuild: X contains infinite or NaN values!");
            alglib.ap.assert(apserv.isfinitevector(y, n), "PolynomialBuild: Y contains infinite or NaN values!");
            tsort.tagsortfastr(ref x, ref y, ref sortrbuf, ref sortrbuf2, n);
            alglib.ap.assert(apserv.aredistinct(x, n), "PolynomialBuild: at least two consequent points are too close!");
            
            //
            // calculate W[j]
            // multi-pass algorithm is used to avoid overflow
            //
            w = new double[n];
            a = x[0];
            b = x[0];
            for(j=0; j<=n-1; j++)
            {
                w[j] = 1;
                a = Math.Min(a, x[j]);
                b = Math.Max(b, x[j]);
            }
            for(k=0; k<=n-1; k++)
            {
                
                //
                // W[K] is used instead of 0.0 because
                // cycle on J does not touch K-th element
                // and we MUST get maximum from ALL elements
                //
                mx = Math.Abs(w[k]);
                for(j=0; j<=n-1; j++)
                {
                    if( j!=k )
                    {
                        v = (b-a)/(x[j]-x[k]);
                        w[j] = w[j]*v;
                        mx = Math.Max(mx, Math.Abs(w[j]));
                    }
                }
                if( k%5==0 )
                {
                    
                    //
                    // every 5-th run we renormalize W[]
                    //
                    v = 1/mx;
                    for(i_=0; i_<=n-1;i_++)
                    {
                        w[i_] = v*w[i_];
                    }
                }
            }
            ratint.barycentricbuildxyw(x, y, w, n, p);
        }


        /*************************************************************************
        Lagrange intepolant: generation of the model on equidistant grid.
        This function has O(N) complexity.

        INPUT PARAMETERS:
            A   -   left boundary of [A,B]
            B   -   right boundary of [A,B]
            Y   -   function values at the nodes, array[0..N-1]
            N   -   number of points, N>=1
                    for N=1 a constant model is constructed.

        OUTPUT PARAMETERS
            P   -   barycentric model which represents Lagrange interpolant
                    (see ratint unit info and BarycentricCalc() description for
                    more information).

          -- ALGLIB --
             Copyright 03.12.2009 by Bochkanov Sergey
        *************************************************************************/
        public static void polynomialbuildeqdist(double a,
            double b,
            double[] y,
            int n,
            ratint.barycentricinterpolant p)
        {
            int i = 0;
            double[] w = new double[0];
            double[] x = new double[0];
            double v = 0;

            alglib.ap.assert(n>0, "PolynomialBuildEqDist: N<=0!");
            alglib.ap.assert(alglib.ap.len(y)>=n, "PolynomialBuildEqDist: Length(Y)<N!");
            alglib.ap.assert(math.isfinite(a), "PolynomialBuildEqDist: A is infinite or NaN!");
            alglib.ap.assert(math.isfinite(b), "PolynomialBuildEqDist: B is infinite or NaN!");
            alglib.ap.assert(apserv.isfinitevector(y, n), "PolynomialBuildEqDist: Y contains infinite or NaN values!");
            alglib.ap.assert((double)(b)!=(double)(a), "PolynomialBuildEqDist: B=A!");
            alglib.ap.assert((double)(a+(b-a)/n)!=(double)(a), "PolynomialBuildEqDist: B is too close to A!");
            
            //
            // Special case: N=1
            //
            if( n==1 )
            {
                x = new double[1];
                w = new double[1];
                x[0] = 0.5*(b+a);
                w[0] = 1;
                ratint.barycentricbuildxyw(x, y, w, 1, p);
                return;
            }
            
            //
            // general case
            //
            x = new double[n];
            w = new double[n];
            v = 1;
            for(i=0; i<=n-1; i++)
            {
                w[i] = v;
                x[i] = a+(b-a)*i/(n-1);
                v = -(v*(n-1-i));
                v = v/(i+1);
            }
            ratint.barycentricbuildxyw(x, y, w, n, p);
        }


        /*************************************************************************
        Lagrange intepolant on Chebyshev grid (first kind).
        This function has O(N) complexity.

        INPUT PARAMETERS:
            A   -   left boundary of [A,B]
            B   -   right boundary of [A,B]
            Y   -   function values at the nodes, array[0..N-1],
                    Y[I] = Y(0.5*(B+A) + 0.5*(B-A)*Cos(PI*(2*i+1)/(2*n)))
            N   -   number of points, N>=1
                    for N=1 a constant model is constructed.

        OUTPUT PARAMETERS
            P   -   barycentric model which represents Lagrange interpolant
                    (see ratint unit info and BarycentricCalc() description for
                    more information).

          -- ALGLIB --
             Copyright 03.12.2009 by Bochkanov Sergey
        *************************************************************************/
        public static void polynomialbuildcheb1(double a,
            double b,
            double[] y,
            int n,
            ratint.barycentricinterpolant p)
        {
            int i = 0;
            double[] w = new double[0];
            double[] x = new double[0];
            double v = 0;
            double t = 0;

            alglib.ap.assert(n>0, "PolynomialBuildCheb1: N<=0!");
            alglib.ap.assert(alglib.ap.len(y)>=n, "PolynomialBuildCheb1: Length(Y)<N!");
            alglib.ap.assert(math.isfinite(a), "PolynomialBuildCheb1: A is infinite or NaN!");
            alglib.ap.assert(math.isfinite(b), "PolynomialBuildCheb1: B is infinite or NaN!");
            alglib.ap.assert(apserv.isfinitevector(y, n), "PolynomialBuildCheb1: Y contains infinite or NaN values!");
            alglib.ap.assert((double)(b)!=(double)(a), "PolynomialBuildCheb1: B=A!");
            
            //
            // Special case: N=1
            //
            if( n==1 )
            {
                x = new double[1];
                w = new double[1];
                x[0] = 0.5*(b+a);
                w[0] = 1;
                ratint.barycentricbuildxyw(x, y, w, 1, p);
                return;
            }
            
            //
            // general case
            //
            x = new double[n];
            w = new double[n];
            v = 1;
            for(i=0; i<=n-1; i++)
            {
                t = Math.Tan(0.5*Math.PI*(2*i+1)/(2*n));
                w[i] = 2*v*t/(1+math.sqr(t));
                x[i] = 0.5*(b+a)+0.5*(b-a)*(1-math.sqr(t))/(1+math.sqr(t));
                v = -v;
            }
            ratint.barycentricbuildxyw(x, y, w, n, p);
        }


        /*************************************************************************
        Lagrange intepolant on Chebyshev grid (second kind).
        This function has O(N) complexity.

        INPUT PARAMETERS:
            A   -   left boundary of [A,B]
            B   -   right boundary of [A,B]
            Y   -   function values at the nodes, array[0..N-1],
                    Y[I] = Y(0.5*(B+A) + 0.5*(B-A)*Cos(PI*i/(n-1)))
            N   -   number of points, N>=1
                    for N=1 a constant model is constructed.

        OUTPUT PARAMETERS
            P   -   barycentric model which represents Lagrange interpolant
                    (see ratint unit info and BarycentricCalc() description for
                    more information).

          -- ALGLIB --
             Copyright 03.12.2009 by Bochkanov Sergey
        *************************************************************************/
        public static void polynomialbuildcheb2(double a,
            double b,
            double[] y,
            int n,
            ratint.barycentricinterpolant p)
        {
            int i = 0;
            double[] w = new double[0];
            double[] x = new double[0];
            double v = 0;

            alglib.ap.assert(n>0, "PolynomialBuildCheb2: N<=0!");
            alglib.ap.assert(alglib.ap.len(y)>=n, "PolynomialBuildCheb2: Length(Y)<N!");
            alglib.ap.assert(math.isfinite(a), "PolynomialBuildCheb2: A is infinite or NaN!");
            alglib.ap.assert(math.isfinite(b), "PolynomialBuildCheb2: B is infinite or NaN!");
            alglib.ap.assert((double)(b)!=(double)(a), "PolynomialBuildCheb2: B=A!");
            alglib.ap.assert(apserv.isfinitevector(y, n), "PolynomialBuildCheb2: Y contains infinite or NaN values!");
            
            //
            // Special case: N=1
            //
            if( n==1 )
            {
                x = new double[1];
                w = new double[1];
                x[0] = 0.5*(b+a);
                w[0] = 1;
                ratint.barycentricbuildxyw(x, y, w, 1, p);
                return;
            }
            
            //
            // general case
            //
            x = new double[n];
            w = new double[n];
            v = 1;
            for(i=0; i<=n-1; i++)
            {
                if( i==0 || i==n-1 )
                {
                    w[i] = v*0.5;
                }
                else
                {
                    w[i] = v;
                }
                x[i] = 0.5*(b+a)+0.5*(b-a)*Math.Cos(Math.PI*i/(n-1));
                v = -v;
            }
            ratint.barycentricbuildxyw(x, y, w, n, p);
        }


        /*************************************************************************
        Fast equidistant polynomial interpolation function with O(N) complexity

        INPUT PARAMETERS:
            A   -   left boundary of [A,B]
            B   -   right boundary of [A,B]
            F   -   function values, array[0..N-1]
            N   -   number of points on equidistant grid, N>=1
                    for N=1 a constant model is constructed.
            T   -   position where P(x) is calculated

        RESULT
            value of the Lagrange interpolant at T
            
        IMPORTANT
            this function provides fast interface which is not overflow-safe
            nor it is very precise.
            the best option is to use  PolynomialBuildEqDist()/BarycentricCalc()
            subroutines unless you are pretty sure that your data will not result
            in overflow.

          -- ALGLIB --
             Copyright 02.12.2009 by Bochkanov Sergey
        *************************************************************************/
        public static double polynomialcalceqdist(double a,
            double b,
            double[] f,
            int n,
            double t)
        {
            double result = 0;
            double s1 = 0;
            double s2 = 0;
            double v = 0;
            double threshold = 0;
            double s = 0;
            double h = 0;
            int i = 0;
            int j = 0;
            double w = 0;
            double x = 0;

            alglib.ap.assert(n>0, "PolynomialCalcEqDist: N<=0!");
            alglib.ap.assert(alglib.ap.len(f)>=n, "PolynomialCalcEqDist: Length(F)<N!");
            alglib.ap.assert(math.isfinite(a), "PolynomialCalcEqDist: A is infinite or NaN!");
            alglib.ap.assert(math.isfinite(b), "PolynomialCalcEqDist: B is infinite or NaN!");
            alglib.ap.assert(apserv.isfinitevector(f, n), "PolynomialCalcEqDist: F contains infinite or NaN values!");
            alglib.ap.assert((double)(b)!=(double)(a), "PolynomialCalcEqDist: B=A!");
            alglib.ap.assert(!Double.IsInfinity(t), "PolynomialCalcEqDist: T is infinite!");
            
            //
            // Special case: T is NAN
            //
            if( Double.IsNaN(t) )
            {
                result = Double.NaN;
                return result;
            }
            
            //
            // Special case: N=1
            //
            if( n==1 )
            {
                result = f[0];
                return result;
            }
            
            //
            // First, decide: should we use "safe" formula (guarded
            // against overflow) or fast one?
            //
            threshold = Math.Sqrt(math.minrealnumber);
            j = 0;
            s = t-a;
            for(i=1; i<=n-1; i++)
            {
                x = a+(double)i/(double)(n-1)*(b-a);
                if( (double)(Math.Abs(t-x))<(double)(Math.Abs(s)) )
                {
                    s = t-x;
                    j = i;
                }
            }
            if( (double)(s)==(double)(0) )
            {
                result = f[j];
                return result;
            }
            if( (double)(Math.Abs(s))>(double)(threshold) )
            {
                
                //
                // use fast formula
                //
                j = -1;
                s = 1.0;
            }
            
            //
            // Calculate using safe or fast barycentric formula
            //
            s1 = 0;
            s2 = 0;
            w = 1.0;
            h = (b-a)/(n-1);
            for(i=0; i<=n-1; i++)
            {
                if( i!=j )
                {
                    v = s*w/(t-(a+i*h));
                    s1 = s1+v*f[i];
                    s2 = s2+v;
                }
                else
                {
                    v = w;
                    s1 = s1+v*f[i];
                    s2 = s2+v;
                }
                w = -(w*(n-1-i));
                w = w/(i+1);
            }
            result = s1/s2;
            return result;
        }


        /*************************************************************************
        Fast polynomial interpolation function on Chebyshev points (first kind)
        with O(N) complexity.

        INPUT PARAMETERS:
            A   -   left boundary of [A,B]
            B   -   right boundary of [A,B]
            F   -   function values, array[0..N-1]
            N   -   number of points on Chebyshev grid (first kind),
                    X[i] = 0.5*(B+A) + 0.5*(B-A)*Cos(PI*(2*i+1)/(2*n))
                    for N=1 a constant model is constructed.
            T   -   position where P(x) is calculated

        RESULT
            value of the Lagrange interpolant at T

        IMPORTANT
            this function provides fast interface which is not overflow-safe
            nor it is very precise.
            the best option is to use  PolIntBuildCheb1()/BarycentricCalc()
            subroutines unless you are pretty sure that your data will not result
            in overflow.

          -- ALGLIB --
             Copyright 02.12.2009 by Bochkanov Sergey
        *************************************************************************/
        public static double polynomialcalccheb1(double a,
            double b,
            double[] f,
            int n,
            double t)
        {
            double result = 0;
            double s1 = 0;
            double s2 = 0;
            double v = 0;
            double threshold = 0;
            double s = 0;
            int i = 0;
            int j = 0;
            double a0 = 0;
            double delta = 0;
            double alpha = 0;
            double beta = 0;
            double ca = 0;
            double sa = 0;
            double tempc = 0;
            double temps = 0;
            double x = 0;
            double w = 0;
            double p1 = 0;

            alglib.ap.assert(n>0, "PolynomialCalcCheb1: N<=0!");
            alglib.ap.assert(alglib.ap.len(f)>=n, "PolynomialCalcCheb1: Length(F)<N!");
            alglib.ap.assert(math.isfinite(a), "PolynomialCalcCheb1: A is infinite or NaN!");
            alglib.ap.assert(math.isfinite(b), "PolynomialCalcCheb1: B is infinite or NaN!");
            alglib.ap.assert(apserv.isfinitevector(f, n), "PolynomialCalcCheb1: F contains infinite or NaN values!");
            alglib.ap.assert((double)(b)!=(double)(a), "PolynomialCalcCheb1: B=A!");
            alglib.ap.assert(!Double.IsInfinity(t), "PolynomialCalcCheb1: T is infinite!");
            
            //
            // Special case: T is NAN
            //
            if( Double.IsNaN(t) )
            {
                result = Double.NaN;
                return result;
            }
            
            //
            // Special case: N=1
            //
            if( n==1 )
            {
                result = f[0];
                return result;
            }
            
            //
            // Prepare information for the recurrence formula
            // used to calculate sin(pi*(2j+1)/(2n+2)) and
            // cos(pi*(2j+1)/(2n+2)):
            //
            // A0    = pi/(2n+2)
            // Delta = pi/(n+1)
            // Alpha = 2 sin^2 (Delta/2)
            // Beta  = sin(Delta)
            //
            // so that sin(..) = sin(A0+j*delta) and cos(..) = cos(A0+j*delta).
            // Then we use
            //
            // sin(x+delta) = sin(x) - (alpha*sin(x) - beta*cos(x))
            // cos(x+delta) = cos(x) - (alpha*cos(x) - beta*sin(x))
            //
            // to repeatedly calculate sin(..) and cos(..).
            //
            threshold = Math.Sqrt(math.minrealnumber);
            t = (t-0.5*(a+b))/(0.5*(b-a));
            a0 = Math.PI/(2*(n-1)+2);
            delta = 2*Math.PI/(2*(n-1)+2);
            alpha = 2*math.sqr(Math.Sin(delta/2));
            beta = Math.Sin(delta);
            
            //
            // First, decide: should we use "safe" formula (guarded
            // against overflow) or fast one?
            //
            ca = Math.Cos(a0);
            sa = Math.Sin(a0);
            j = 0;
            x = ca;
            s = t-x;
            for(i=1; i<=n-1; i++)
            {
                
                //
                // Next X[i]
                //
                temps = sa-(alpha*sa-beta*ca);
                tempc = ca-(alpha*ca+beta*sa);
                sa = temps;
                ca = tempc;
                x = ca;
                
                //
                // Use X[i]
                //
                if( (double)(Math.Abs(t-x))<(double)(Math.Abs(s)) )
                {
                    s = t-x;
                    j = i;
                }
            }
            if( (double)(s)==(double)(0) )
            {
                result = f[j];
                return result;
            }
            if( (double)(Math.Abs(s))>(double)(threshold) )
            {
                
                //
                // use fast formula
                //
                j = -1;
                s = 1.0;
            }
            
            //
            // Calculate using safe or fast barycentric formula
            //
            s1 = 0;
            s2 = 0;
            ca = Math.Cos(a0);
            sa = Math.Sin(a0);
            p1 = 1.0;
            for(i=0; i<=n-1; i++)
            {
                
                //
                // Calculate X[i], W[i]
                //
                x = ca;
                w = p1*sa;
                
                //
                // Proceed
                //
                if( i!=j )
                {
                    v = s*w/(t-x);
                    s1 = s1+v*f[i];
                    s2 = s2+v;
                }
                else
                {
                    v = w;
                    s1 = s1+v*f[i];
                    s2 = s2+v;
                }
                
                //
                // Next CA, SA, P1
                //
                temps = sa-(alpha*sa-beta*ca);
                tempc = ca-(alpha*ca+beta*sa);
                sa = temps;
                ca = tempc;
                p1 = -p1;
            }
            result = s1/s2;
            return result;
        }


        /*************************************************************************
        Fast polynomial interpolation function on Chebyshev points (second kind)
        with O(N) complexity.

        INPUT PARAMETERS:
            A   -   left boundary of [A,B]
            B   -   right boundary of [A,B]
            F   -   function values, array[0..N-1]
            N   -   number of points on Chebyshev grid (second kind),
                    X[i] = 0.5*(B+A) + 0.5*(B-A)*Cos(PI*i/(n-1))
                    for N=1 a constant model is constructed.
            T   -   position where P(x) is calculated

        RESULT
            value of the Lagrange interpolant at T

        IMPORTANT
            this function provides fast interface which is not overflow-safe
            nor it is very precise.
            the best option is to use PolIntBuildCheb2()/BarycentricCalc()
            subroutines unless you are pretty sure that your data will not result
            in overflow.

          -- ALGLIB --
             Copyright 02.12.2009 by Bochkanov Sergey
        *************************************************************************/
        public static double polynomialcalccheb2(double a,
            double b,
            double[] f,
            int n,
            double t)
        {
            double result = 0;
            double s1 = 0;
            double s2 = 0;
            double v = 0;
            double threshold = 0;
            double s = 0;
            int i = 0;
            int j = 0;
            double a0 = 0;
            double delta = 0;
            double alpha = 0;
            double beta = 0;
            double ca = 0;
            double sa = 0;
            double tempc = 0;
            double temps = 0;
            double x = 0;
            double w = 0;
            double p1 = 0;

            alglib.ap.assert(n>0, "PolynomialCalcCheb2: N<=0!");
            alglib.ap.assert(alglib.ap.len(f)>=n, "PolynomialCalcCheb2: Length(F)<N!");
            alglib.ap.assert(math.isfinite(a), "PolynomialCalcCheb2: A is infinite or NaN!");
            alglib.ap.assert(math.isfinite(b), "PolynomialCalcCheb2: B is infinite or NaN!");
            alglib.ap.assert((double)(b)!=(double)(a), "PolynomialCalcCheb2: B=A!");
            alglib.ap.assert(apserv.isfinitevector(f, n), "PolynomialCalcCheb2: F contains infinite or NaN values!");
            alglib.ap.assert(!Double.IsInfinity(t), "PolynomialCalcEqDist: T is infinite!");
            
            //
            // Special case: T is NAN
            //
            if( Double.IsNaN(t) )
            {
                result = Double.NaN;
                return result;
            }
            
            //
            // Special case: N=1
            //
            if( n==1 )
            {
                result = f[0];
                return result;
            }
            
            //
            // Prepare information for the recurrence formula
            // used to calculate sin(pi*i/n) and
            // cos(pi*i/n):
            //
            // A0    = 0
            // Delta = pi/n
            // Alpha = 2 sin^2 (Delta/2)
            // Beta  = sin(Delta)
            //
            // so that sin(..) = sin(A0+j*delta) and cos(..) = cos(A0+j*delta).
            // Then we use
            //
            // sin(x+delta) = sin(x) - (alpha*sin(x) - beta*cos(x))
            // cos(x+delta) = cos(x) - (alpha*cos(x) - beta*sin(x))
            //
            // to repeatedly calculate sin(..) and cos(..).
            //
            threshold = Math.Sqrt(math.minrealnumber);
            t = (t-0.5*(a+b))/(0.5*(b-a));
            a0 = 0.0;
            delta = Math.PI/(n-1);
            alpha = 2*math.sqr(Math.Sin(delta/2));
            beta = Math.Sin(delta);
            
            //
            // First, decide: should we use "safe" formula (guarded
            // against overflow) or fast one?
            //
            ca = Math.Cos(a0);
            sa = Math.Sin(a0);
            j = 0;
            x = ca;
            s = t-x;
            for(i=1; i<=n-1; i++)
            {
                
                //
                // Next X[i]
                //
                temps = sa-(alpha*sa-beta*ca);
                tempc = ca-(alpha*ca+beta*sa);
                sa = temps;
                ca = tempc;
                x = ca;
                
                //
                // Use X[i]
                //
                if( (double)(Math.Abs(t-x))<(double)(Math.Abs(s)) )
                {
                    s = t-x;
                    j = i;
                }
            }
            if( (double)(s)==(double)(0) )
            {
                result = f[j];
                return result;
            }
            if( (double)(Math.Abs(s))>(double)(threshold) )
            {
                
                //
                // use fast formula
                //
                j = -1;
                s = 1.0;
            }
            
            //
            // Calculate using safe or fast barycentric formula
            //
            s1 = 0;
            s2 = 0;
            ca = Math.Cos(a0);
            sa = Math.Sin(a0);
            p1 = 1.0;
            for(i=0; i<=n-1; i++)
            {
                
                //
                // Calculate X[i], W[i]
                //
                x = ca;
                if( i==0 || i==n-1 )
                {
                    w = 0.5*p1;
                }
                else
                {
                    w = 1.0*p1;
                }
                
                //
                // Proceed
                //
                if( i!=j )
                {
                    v = s*w/(t-x);
                    s1 = s1+v*f[i];
                    s2 = s2+v;
                }
                else
                {
                    v = w;
                    s1 = s1+v*f[i];
                    s2 = s2+v;
                }
                
                //
                // Next CA, SA, P1
                //
                temps = sa-(alpha*sa-beta*ca);
                tempc = ca-(alpha*ca+beta*sa);
                sa = temps;
                ca = tempc;
                p1 = -p1;
            }
            result = s1/s2;
            return result;
        }


    }
    public class spline1d
    {
        /*************************************************************************
        1-dimensional spline interpolant
        *************************************************************************/
        public class spline1dinterpolant
        {
            public bool periodic;
            public int n;
            public int k;
            public int continuity;
            public double[] x;
            public double[] c;
            public spline1dinterpolant()
            {
                x = new double[0];
                c = new double[0];
            }
        };




        /*************************************************************************
        This subroutine builds linear spline interpolant

        INPUT PARAMETERS:
            X   -   spline nodes, array[0..N-1]
            Y   -   function values, array[0..N-1]
            N   -   points count (optional):
                    * N>=2
                    * if given, only first N points are used to build spline
                    * if not given, automatically detected from X/Y sizes
                      (len(X) must be equal to len(Y))
            
        OUTPUT PARAMETERS:
            C   -   spline interpolant


        ORDER OF POINTS

        Subroutine automatically sorts points, so caller may pass unsorted array.

          -- ALGLIB PROJECT --
             Copyright 24.06.2007 by Bochkanov Sergey
        *************************************************************************/
        public static void spline1dbuildlinear(double[] x,
            double[] y,
            int n,
            spline1dinterpolant c)
        {
            int i = 0;

            x = (double[])x.Clone();
            y = (double[])y.Clone();

            alglib.ap.assert(n>1, "Spline1DBuildLinear: N<2!");
            alglib.ap.assert(alglib.ap.len(x)>=n, "Spline1DBuildLinear: Length(X)<N!");
            alglib.ap.assert(alglib.ap.len(y)>=n, "Spline1DBuildLinear: Length(Y)<N!");
            
            //
            // check and sort points
            //
            alglib.ap.assert(apserv.isfinitevector(x, n), "Spline1DBuildLinear: X contains infinite or NAN values!");
            alglib.ap.assert(apserv.isfinitevector(y, n), "Spline1DBuildLinear: Y contains infinite or NAN values!");
            heapsortpoints(ref x, ref y, n);
            alglib.ap.assert(apserv.aredistinct(x, n), "Spline1DBuildLinear: at least two consequent points are too close!");
            
            //
            // Build
            //
            c.periodic = false;
            c.n = n;
            c.k = 3;
            c.continuity = 0;
            c.x = new double[n];
            c.c = new double[4*(n-1)+2];
            for(i=0; i<=n-1; i++)
            {
                c.x[i] = x[i];
            }
            for(i=0; i<=n-2; i++)
            {
                c.c[4*i+0] = y[i];
                c.c[4*i+1] = (y[i+1]-y[i])/(x[i+1]-x[i]);
                c.c[4*i+2] = 0;
                c.c[4*i+3] = 0;
            }
            c.c[4*(n-1)+0] = y[n-1];
            c.c[4*(n-1)+1] = c.c[4*(n-2)+1];
        }


        /*************************************************************************
        This subroutine builds cubic spline interpolant.

        INPUT PARAMETERS:
            X           -   spline nodes, array[0..N-1].
            Y           -   function values, array[0..N-1].
            
        OPTIONAL PARAMETERS:
            N           -   points count:
                            * N>=2
                            * if given, only first N points are used to build spline
                            * if not given, automatically detected from X/Y sizes
                              (len(X) must be equal to len(Y))
            BoundLType  -   boundary condition type for the left boundary
            BoundL      -   left boundary condition (first or second derivative,
                            depending on the BoundLType)
            BoundRType  -   boundary condition type for the right boundary
            BoundR      -   right boundary condition (first or second derivative,
                            depending on the BoundRType)

        OUTPUT PARAMETERS:
            C           -   spline interpolant

        ORDER OF POINTS

        Subroutine automatically sorts points, so caller may pass unsorted array.

        SETTING BOUNDARY VALUES:

        The BoundLType/BoundRType parameters can have the following values:
            * -1, which corresonds to the periodic (cyclic) boundary conditions.
                  In this case:
                  * both BoundLType and BoundRType must be equal to -1.
                  * BoundL/BoundR are ignored
                  * Y[last] is ignored (it is assumed to be equal to Y[first]).
            *  0, which  corresponds  to  the  parabolically   terminated  spline
                  (BoundL and/or BoundR are ignored).
            *  1, which corresponds to the first derivative boundary condition
            *  2, which corresponds to the second derivative boundary condition
            *  by default, BoundType=0 is used

        PROBLEMS WITH PERIODIC BOUNDARY CONDITIONS:

        Problems with periodic boundary conditions have Y[first_point]=Y[last_point].
        However, this subroutine doesn't require you to specify equal  values  for
        the first and last points - it automatically forces them  to  be  equal by
        copying  Y[first_point]  (corresponds  to the leftmost,  minimal  X[])  to
        Y[last_point]. However it is recommended to pass consistent values of Y[],
        i.e. to make Y[first_point]=Y[last_point].

          -- ALGLIB PROJECT --
             Copyright 23.06.2007 by Bochkanov Sergey
        *************************************************************************/
        public static void spline1dbuildcubic(double[] x,
            double[] y,
            int n,
            int boundltype,
            double boundl,
            int boundrtype,
            double boundr,
            spline1dinterpolant c)
        {
            double[] a1 = new double[0];
            double[] a2 = new double[0];
            double[] a3 = new double[0];
            double[] b = new double[0];
            double[] dt = new double[0];
            double[] d = new double[0];
            int[] p = new int[0];
            int ylen = 0;

            x = (double[])x.Clone();
            y = (double[])y.Clone();

            
            //
            // check correctness of boundary conditions
            //
            alglib.ap.assert(((boundltype==-1 || boundltype==0) || boundltype==1) || boundltype==2, "Spline1DBuildCubic: incorrect BoundLType!");
            alglib.ap.assert(((boundrtype==-1 || boundrtype==0) || boundrtype==1) || boundrtype==2, "Spline1DBuildCubic: incorrect BoundRType!");
            alglib.ap.assert((boundrtype==-1 && boundltype==-1) || (boundrtype!=-1 && boundltype!=-1), "Spline1DBuildCubic: incorrect BoundLType/BoundRType!");
            if( boundltype==1 || boundltype==2 )
            {
                alglib.ap.assert(math.isfinite(boundl), "Spline1DBuildCubic: BoundL is infinite or NAN!");
            }
            if( boundrtype==1 || boundrtype==2 )
            {
                alglib.ap.assert(math.isfinite(boundr), "Spline1DBuildCubic: BoundR is infinite or NAN!");
            }
            
            //
            // check lengths of arguments
            //
            alglib.ap.assert(n>=2, "Spline1DBuildCubic: N<2!");
            alglib.ap.assert(alglib.ap.len(x)>=n, "Spline1DBuildCubic: Length(X)<N!");
            alglib.ap.assert(alglib.ap.len(y)>=n, "Spline1DBuildCubic: Length(Y)<N!");
            
            //
            // check and sort points
            //
            ylen = n;
            if( boundltype==-1 )
            {
                ylen = n-1;
            }
            alglib.ap.assert(apserv.isfinitevector(x, n), "Spline1DBuildCubic: X contains infinite or NAN values!");
            alglib.ap.assert(apserv.isfinitevector(y, ylen), "Spline1DBuildCubic: Y contains infinite or NAN values!");
            heapsortppoints(ref x, ref y, ref p, n);
            alglib.ap.assert(apserv.aredistinct(x, n), "Spline1DBuildCubic: at least two consequent points are too close!");
            
            //
            // Now we've checked and preordered everything,
            // so we can call internal function to calculate derivatives,
            // and then build Hermite spline using these derivatives
            //
            if( boundltype==-1 || boundrtype==-1 )
            {
                y[n-1] = y[0];
            }
            spline1dgriddiffcubicinternal(x, ref y, n, boundltype, boundl, boundrtype, boundr, ref d, ref a1, ref a2, ref a3, ref b, ref dt);
            spline1dbuildhermite(x, y, d, n, c);
            c.periodic = boundltype==-1 || boundrtype==-1;
            c.continuity = 2;
        }


        /*************************************************************************
        This function solves following problem: given table y[] of function values
        at nodes x[], it calculates and returns table of function derivatives  d[]
        (calculated at the same nodes x[]).

        This function yields same result as Spline1DBuildCubic() call followed  by
        sequence of Spline1DDiff() calls, but it can be several times faster  when
        called for ordered X[] and X2[].

        INPUT PARAMETERS:
            X           -   spline nodes
            Y           -   function values

        OPTIONAL PARAMETERS:
            N           -   points count:
                            * N>=2
                            * if given, only first N points are used
                            * if not given, automatically detected from X/Y sizes
                              (len(X) must be equal to len(Y))
            BoundLType  -   boundary condition type for the left boundary
            BoundL      -   left boundary condition (first or second derivative,
                            depending on the BoundLType)
            BoundRType  -   boundary condition type for the right boundary
            BoundR      -   right boundary condition (first or second derivative,
                            depending on the BoundRType)

        OUTPUT PARAMETERS:
            D           -   derivative values at X[]

        ORDER OF POINTS

        Subroutine automatically sorts points, so caller may pass unsorted array.
        Derivative values are correctly reordered on return, so  D[I]  is  always
        equal to S'(X[I]) independently of points order.

        SETTING BOUNDARY VALUES:

        The BoundLType/BoundRType parameters can have the following values:
            * -1, which corresonds to the periodic (cyclic) boundary conditions.
                  In this case:
                  * both BoundLType and BoundRType must be equal to -1.
                  * BoundL/BoundR are ignored
                  * Y[last] is ignored (it is assumed to be equal to Y[first]).
            *  0, which  corresponds  to  the  parabolically   terminated  spline
                  (BoundL and/or BoundR are ignored).
            *  1, which corresponds to the first derivative boundary condition
            *  2, which corresponds to the second derivative boundary condition
            *  by default, BoundType=0 is used

        PROBLEMS WITH PERIODIC BOUNDARY CONDITIONS:

        Problems with periodic boundary conditions have Y[first_point]=Y[last_point].
        However, this subroutine doesn't require you to specify equal  values  for
        the first and last points - it automatically forces them  to  be  equal by
        copying  Y[first_point]  (corresponds  to the leftmost,  minimal  X[])  to
        Y[last_point]. However it is recommended to pass consistent values of Y[],
        i.e. to make Y[first_point]=Y[last_point].

          -- ALGLIB PROJECT --
             Copyright 03.09.2010 by Bochkanov Sergey
        *************************************************************************/
        public static void spline1dgriddiffcubic(double[] x,
            double[] y,
            int n,
            int boundltype,
            double boundl,
            int boundrtype,
            double boundr,
            ref double[] d)
        {
            double[] a1 = new double[0];
            double[] a2 = new double[0];
            double[] a3 = new double[0];
            double[] b = new double[0];
            double[] dt = new double[0];
            int[] p = new int[0];
            int i = 0;
            int ylen = 0;
            int i_ = 0;

            x = (double[])x.Clone();
            y = (double[])y.Clone();
            d = new double[0];

            
            //
            // check correctness of boundary conditions
            //
            alglib.ap.assert(((boundltype==-1 || boundltype==0) || boundltype==1) || boundltype==2, "Spline1DGridDiffCubic: incorrect BoundLType!");
            alglib.ap.assert(((boundrtype==-1 || boundrtype==0) || boundrtype==1) || boundrtype==2, "Spline1DGridDiffCubic: incorrect BoundRType!");
            alglib.ap.assert((boundrtype==-1 && boundltype==-1) || (boundrtype!=-1 && boundltype!=-1), "Spline1DGridDiffCubic: incorrect BoundLType/BoundRType!");
            if( boundltype==1 || boundltype==2 )
            {
                alglib.ap.assert(math.isfinite(boundl), "Spline1DGridDiffCubic: BoundL is infinite or NAN!");
            }
            if( boundrtype==1 || boundrtype==2 )
            {
                alglib.ap.assert(math.isfinite(boundr), "Spline1DGridDiffCubic: BoundR is infinite or NAN!");
            }
            
            //
            // check lengths of arguments
            //
            alglib.ap.assert(n>=2, "Spline1DGridDiffCubic: N<2!");
            alglib.ap.assert(alglib.ap.len(x)>=n, "Spline1DGridDiffCubic: Length(X)<N!");
            alglib.ap.assert(alglib.ap.len(y)>=n, "Spline1DGridDiffCubic: Length(Y)<N!");
            
            //
            // check and sort points
            //
            ylen = n;
            if( boundltype==-1 )
            {
                ylen = n-1;
            }
            alglib.ap.assert(apserv.isfinitevector(x, n), "Spline1DGridDiffCubic: X contains infinite or NAN values!");
            alglib.ap.assert(apserv.isfinitevector(y, ylen), "Spline1DGridDiffCubic: Y contains infinite or NAN values!");
            heapsortppoints(ref x, ref y, ref p, n);
            alglib.ap.assert(apserv.aredistinct(x, n), "Spline1DGridDiffCubic: at least two consequent points are too close!");
            
            //
            // Now we've checked and preordered everything,
            // so we can call internal function.
            //
            spline1dgriddiffcubicinternal(x, ref y, n, boundltype, boundl, boundrtype, boundr, ref d, ref a1, ref a2, ref a3, ref b, ref dt);
            
            //
            // Remember that HeapSortPPoints() call?
            // Now we have to reorder them back.
            //
            if( alglib.ap.len(dt)<n )
            {
                dt = new double[n];
            }
            for(i=0; i<=n-1; i++)
            {
                dt[p[i]] = d[i];
            }
            for(i_=0; i_<=n-1;i_++)
            {
                d[i_] = dt[i_];
            }
        }


        /*************************************************************************
        This function solves following problem: given table y[] of function values
        at  nodes  x[],  it  calculates  and  returns  tables  of first and second
        function derivatives d1[] and d2[] (calculated at the same nodes x[]).

        This function yields same result as Spline1DBuildCubic() call followed  by
        sequence of Spline1DDiff() calls, but it can be several times faster  when
        called for ordered X[] and X2[].

        INPUT PARAMETERS:
            X           -   spline nodes
            Y           -   function values

        OPTIONAL PARAMETERS:
            N           -   points count:
                            * N>=2
                            * if given, only first N points are used
                            * if not given, automatically detected from X/Y sizes
                              (len(X) must be equal to len(Y))
            BoundLType  -   boundary condition type for the left boundary
            BoundL      -   left boundary condition (first or second derivative,
                            depending on the BoundLType)
            BoundRType  -   boundary condition type for the right boundary
            BoundR      -   right boundary condition (first or second derivative,
                            depending on the BoundRType)

        OUTPUT PARAMETERS:
            D1          -   S' values at X[]
            D2          -   S'' values at X[]

        ORDER OF POINTS

        Subroutine automatically sorts points, so caller may pass unsorted array.
        Derivative values are correctly reordered on return, so  D[I]  is  always
        equal to S'(X[I]) independently of points order.

        SETTING BOUNDARY VALUES:

        The BoundLType/BoundRType parameters can have the following values:
            * -1, which corresonds to the periodic (cyclic) boundary conditions.
                  In this case:
                  * both BoundLType and BoundRType must be equal to -1.
                  * BoundL/BoundR are ignored
                  * Y[last] is ignored (it is assumed to be equal to Y[first]).
            *  0, which  corresponds  to  the  parabolically   terminated  spline
                  (BoundL and/or BoundR are ignored).
            *  1, which corresponds to the first derivative boundary condition
            *  2, which corresponds to the second derivative boundary condition
            *  by default, BoundType=0 is used

        PROBLEMS WITH PERIODIC BOUNDARY CONDITIONS:

        Problems with periodic boundary conditions have Y[first_point]=Y[last_point].
        However, this subroutine doesn't require you to specify equal  values  for
        the first and last points - it automatically forces them  to  be  equal by
        copying  Y[first_point]  (corresponds  to the leftmost,  minimal  X[])  to
        Y[last_point]. However it is recommended to pass consistent values of Y[],
        i.e. to make Y[first_point]=Y[last_point].

          -- ALGLIB PROJECT --
             Copyright 03.09.2010 by Bochkanov Sergey
        *************************************************************************/
        public static void spline1dgriddiff2cubic(double[] x,
            double[] y,
            int n,
            int boundltype,
            double boundl,
            int boundrtype,
            double boundr,
            ref double[] d1,
            ref double[] d2)
        {
            double[] a1 = new double[0];
            double[] a2 = new double[0];
            double[] a3 = new double[0];
            double[] b = new double[0];
            double[] dt = new double[0];
            int[] p = new int[0];
            int i = 0;
            int ylen = 0;
            double delta = 0;
            double delta2 = 0;
            double delta3 = 0;
            double s0 = 0;
            double s1 = 0;
            double s2 = 0;
            double s3 = 0;
            int i_ = 0;

            x = (double[])x.Clone();
            y = (double[])y.Clone();
            d1 = new double[0];
            d2 = new double[0];

            
            //
            // check correctness of boundary conditions
            //
            alglib.ap.assert(((boundltype==-1 || boundltype==0) || boundltype==1) || boundltype==2, "Spline1DGridDiff2Cubic: incorrect BoundLType!");
            alglib.ap.assert(((boundrtype==-1 || boundrtype==0) || boundrtype==1) || boundrtype==2, "Spline1DGridDiff2Cubic: incorrect BoundRType!");
            alglib.ap.assert((boundrtype==-1 && boundltype==-1) || (boundrtype!=-1 && boundltype!=-1), "Spline1DGridDiff2Cubic: incorrect BoundLType/BoundRType!");
            if( boundltype==1 || boundltype==2 )
            {
                alglib.ap.assert(math.isfinite(boundl), "Spline1DGridDiff2Cubic: BoundL is infinite or NAN!");
            }
            if( boundrtype==1 || boundrtype==2 )
            {
                alglib.ap.assert(math.isfinite(boundr), "Spline1DGridDiff2Cubic: BoundR is infinite or NAN!");
            }
            
            //
            // check lengths of arguments
            //
            alglib.ap.assert(n>=2, "Spline1DGridDiff2Cubic: N<2!");
            alglib.ap.assert(alglib.ap.len(x)>=n, "Spline1DGridDiff2Cubic: Length(X)<N!");
            alglib.ap.assert(alglib.ap.len(y)>=n, "Spline1DGridDiff2Cubic: Length(Y)<N!");
            
            //
            // check and sort points
            //
            ylen = n;
            if( boundltype==-1 )
            {
                ylen = n-1;
            }
            alglib.ap.assert(apserv.isfinitevector(x, n), "Spline1DGridDiff2Cubic: X contains infinite or NAN values!");
            alglib.ap.assert(apserv.isfinitevector(y, ylen), "Spline1DGridDiff2Cubic: Y contains infinite or NAN values!");
            heapsortppoints(ref x, ref y, ref p, n);
            alglib.ap.assert(apserv.aredistinct(x, n), "Spline1DGridDiff2Cubic: at least two consequent points are too close!");
            
            //
            // Now we've checked and preordered everything,
            // so we can call internal function.
            //
            // After this call we will calculate second derivatives
            // (manually, by converting to the power basis)
            //
            spline1dgriddiffcubicinternal(x, ref y, n, boundltype, boundl, boundrtype, boundr, ref d1, ref a1, ref a2, ref a3, ref b, ref dt);
            d2 = new double[n];
            delta = 0;
            s2 = 0;
            s3 = 0;
            for(i=0; i<=n-2; i++)
            {
                
                //
                // We convert from Hermite basis to the power basis.
                // Si is coefficient before x^i.
                //
                // Inside this cycle we need just S2,
                // because we calculate S'' exactly at spline node,
                // (only x^2 matters at x=0), but after iterations
                // will be over, we will need other coefficients
                // to calculate spline value at the last node.
                //
                delta = x[i+1]-x[i];
                delta2 = math.sqr(delta);
                delta3 = delta*delta2;
                s0 = y[i];
                s1 = d1[i];
                s2 = (3*(y[i+1]-y[i])-2*d1[i]*delta-d1[i+1]*delta)/delta2;
                s3 = (2*(y[i]-y[i+1])+d1[i]*delta+d1[i+1]*delta)/delta3;
                d2[i] = 2*s2;
            }
            d2[n-1] = 2*s2+6*s3*delta;
            
            //
            // Remember that HeapSortPPoints() call?
            // Now we have to reorder them back.
            //
            if( alglib.ap.len(dt)<n )
            {
                dt = new double[n];
            }
            for(i=0; i<=n-1; i++)
            {
                dt[p[i]] = d1[i];
            }
            for(i_=0; i_<=n-1;i_++)
            {
                d1[i_] = dt[i_];
            }
            for(i=0; i<=n-1; i++)
            {
                dt[p[i]] = d2[i];
            }
            for(i_=0; i_<=n-1;i_++)
            {
                d2[i_] = dt[i_];
            }
        }


        /*************************************************************************
        This function solves following problem: given table y[] of function values
        at old nodes x[]  and new nodes  x2[],  it calculates and returns table of
        function values y2[] (calculated at x2[]).

        This function yields same result as Spline1DBuildCubic() call followed  by
        sequence of Spline1DDiff() calls, but it can be several times faster  when
        called for ordered X[] and X2[].

        INPUT PARAMETERS:
            X           -   old spline nodes
            Y           -   function values
            X2           -  new spline nodes

        OPTIONAL PARAMETERS:
            N           -   points count:
                            * N>=2
                            * if given, only first N points from X/Y are used
                            * if not given, automatically detected from X/Y sizes
                              (len(X) must be equal to len(Y))
            BoundLType  -   boundary condition type for the left boundary
            BoundL      -   left boundary condition (first or second derivative,
                            depending on the BoundLType)
            BoundRType  -   boundary condition type for the right boundary
            BoundR      -   right boundary condition (first or second derivative,
                            depending on the BoundRType)
            N2          -   new points count:
                            * N2>=2
                            * if given, only first N2 points from X2 are used
                            * if not given, automatically detected from X2 size

        OUTPUT PARAMETERS:
            F2          -   function values at X2[]

        ORDER OF POINTS

        Subroutine automatically sorts points, so caller  may pass unsorted array.
        Function  values  are correctly reordered on  return, so F2[I]  is  always
        equal to S(X2[I]) independently of points order.

        SETTING BOUNDARY VALUES:

        The BoundLType/BoundRType parameters can have the following values:
            * -1, which corresonds to the periodic (cyclic) boundary conditions.
                  In this case:
                  * both BoundLType and BoundRType must be equal to -1.
                  * BoundL/BoundR are ignored
                  * Y[last] is ignored (it is assumed to be equal to Y[first]).
            *  0, which  corresponds  to  the  parabolically   terminated  spline
                  (BoundL and/or BoundR are ignored).
            *  1, which corresponds to the first derivative boundary condition
            *  2, which corresponds to the second derivative boundary condition
            *  by default, BoundType=0 is used

        PROBLEMS WITH PERIODIC BOUNDARY CONDITIONS:

        Problems with periodic boundary conditions have Y[first_point]=Y[last_point].
        However, this subroutine doesn't require you to specify equal  values  for
        the first and last points - it automatically forces them  to  be  equal by
        copying  Y[first_point]  (corresponds  to the leftmost,  minimal  X[])  to
        Y[last_point]. However it is recommended to pass consistent values of Y[],
        i.e. to make Y[first_point]=Y[last_point].

          -- ALGLIB PROJECT --
             Copyright 03.09.2010 by Bochkanov Sergey
        *************************************************************************/
        public static void spline1dconvcubic(double[] x,
            double[] y,
            int n,
            int boundltype,
            double boundl,
            int boundrtype,
            double boundr,
            double[] x2,
            int n2,
            ref double[] y2)
        {
            double[] a1 = new double[0];
            double[] a2 = new double[0];
            double[] a3 = new double[0];
            double[] b = new double[0];
            double[] d = new double[0];
            double[] dt = new double[0];
            double[] d1 = new double[0];
            double[] d2 = new double[0];
            int[] p = new int[0];
            int[] p2 = new int[0];
            int i = 0;
            int ylen = 0;
            double t = 0;
            double t2 = 0;
            int i_ = 0;

            x = (double[])x.Clone();
            y = (double[])y.Clone();
            x2 = (double[])x2.Clone();
            y2 = new double[0];

            
            //
            // check correctness of boundary conditions
            //
            alglib.ap.assert(((boundltype==-1 || boundltype==0) || boundltype==1) || boundltype==2, "Spline1DConvCubic: incorrect BoundLType!");
            alglib.ap.assert(((boundrtype==-1 || boundrtype==0) || boundrtype==1) || boundrtype==2, "Spline1DConvCubic: incorrect BoundRType!");
            alglib.ap.assert((boundrtype==-1 && boundltype==-1) || (boundrtype!=-1 && boundltype!=-1), "Spline1DConvCubic: incorrect BoundLType/BoundRType!");
            if( boundltype==1 || boundltype==2 )
            {
                alglib.ap.assert(math.isfinite(boundl), "Spline1DConvCubic: BoundL is infinite or NAN!");
            }
            if( boundrtype==1 || boundrtype==2 )
            {
                alglib.ap.assert(math.isfinite(boundr), "Spline1DConvCubic: BoundR is infinite or NAN!");
            }
            
            //
            // check lengths of arguments
            //
            alglib.ap.assert(n>=2, "Spline1DConvCubic: N<2!");
            alglib.ap.assert(alglib.ap.len(x)>=n, "Spline1DConvCubic: Length(X)<N!");
            alglib.ap.assert(alglib.ap.len(y)>=n, "Spline1DConvCubic: Length(Y)<N!");
            alglib.ap.assert(n2>=2, "Spline1DConvCubic: N2<2!");
            alglib.ap.assert(alglib.ap.len(x2)>=n2, "Spline1DConvCubic: Length(X2)<N2!");
            
            //
            // check and sort X/Y
            //
            ylen = n;
            if( boundltype==-1 )
            {
                ylen = n-1;
            }
            alglib.ap.assert(apserv.isfinitevector(x, n), "Spline1DConvCubic: X contains infinite or NAN values!");
            alglib.ap.assert(apserv.isfinitevector(y, ylen), "Spline1DConvCubic: Y contains infinite or NAN values!");
            alglib.ap.assert(apserv.isfinitevector(x2, n2), "Spline1DConvCubic: X2 contains infinite or NAN values!");
            heapsortppoints(ref x, ref y, ref p, n);
            alglib.ap.assert(apserv.aredistinct(x, n), "Spline1DConvCubic: at least two consequent points are too close!");
            
            //
            // set up DT (we will need it below)
            //
            dt = new double[Math.Max(n, n2)];
            
            //
            // sort X2:
            // * use fake array DT because HeapSortPPoints() needs both integer AND real arrays
            // * if we have periodic problem, wrap points
            // * sort them, store permutation at P2
            //
            if( boundrtype==-1 && boundltype==-1 )
            {
                for(i=0; i<=n2-1; i++)
                {
                    t = x2[i];
                    apserv.apperiodicmap(ref t, x[0], x[n-1], ref t2);
                    x2[i] = t;
                }
            }
            heapsortppoints(ref x2, ref dt, ref p2, n2);
            
            //
            // Now we've checked and preordered everything, so we:
            // * call internal GridDiff() function to get Hermite form of spline
            // * convert using internal Conv() function
            // * convert Y2 back to original order
            //
            spline1dgriddiffcubicinternal(x, ref y, n, boundltype, boundl, boundrtype, boundr, ref d, ref a1, ref a2, ref a3, ref b, ref dt);
            spline1dconvdiffinternal(x, y, d, n, x2, n2, ref y2, true, ref d1, false, ref d2, false);
            alglib.ap.assert(alglib.ap.len(dt)>=n2, "Spline1DConvCubic: internal error!");
            for(i=0; i<=n2-1; i++)
            {
                dt[p2[i]] = y2[i];
            }
            for(i_=0; i_<=n2-1;i_++)
            {
                y2[i_] = dt[i_];
            }
        }


        /*************************************************************************
        This function solves following problem: given table y[] of function values
        at old nodes x[]  and new nodes  x2[],  it calculates and returns table of
        function values y2[] and derivatives d2[] (calculated at x2[]).

        This function yields same result as Spline1DBuildCubic() call followed  by
        sequence of Spline1DDiff() calls, but it can be several times faster  when
        called for ordered X[] and X2[].

        INPUT PARAMETERS:
            X           -   old spline nodes
            Y           -   function values
            X2           -  new spline nodes

        OPTIONAL PARAMETERS:
            N           -   points count:
                            * N>=2
                            * if given, only first N points from X/Y are used
                            * if not given, automatically detected from X/Y sizes
                              (len(X) must be equal to len(Y))
            BoundLType  -   boundary condition type for the left boundary
            BoundL      -   left boundary condition (first or second derivative,
                            depending on the BoundLType)
            BoundRType  -   boundary condition type for the right boundary
            BoundR      -   right boundary condition (first or second derivative,
                            depending on the BoundRType)
            N2          -   new points count:
                            * N2>=2
                            * if given, only first N2 points from X2 are used
                            * if not given, automatically detected from X2 size

        OUTPUT PARAMETERS:
            F2          -   function values at X2[]
            D2          -   first derivatives at X2[]

        ORDER OF POINTS

        Subroutine automatically sorts points, so caller  may pass unsorted array.
        Function  values  are correctly reordered on  return, so F2[I]  is  always
        equal to S(X2[I]) independently of points order.

        SETTING BOUNDARY VALUES:

        The BoundLType/BoundRType parameters can have the following values:
            * -1, which corresonds to the periodic (cyclic) boundary conditions.
                  In this case:
                  * both BoundLType and BoundRType must be equal to -1.
                  * BoundL/BoundR are ignored
                  * Y[last] is ignored (it is assumed to be equal to Y[first]).
            *  0, which  corresponds  to  the  parabolically   terminated  spline
                  (BoundL and/or BoundR are ignored).
            *  1, which corresponds to the first derivative boundary condition
            *  2, which corresponds to the second derivative boundary condition
            *  by default, BoundType=0 is used

        PROBLEMS WITH PERIODIC BOUNDARY CONDITIONS:

        Problems with periodic boundary conditions have Y[first_point]=Y[last_point].
        However, this subroutine doesn't require you to specify equal  values  for
        the first and last points - it automatically forces them  to  be  equal by
        copying  Y[first_point]  (corresponds  to the leftmost,  minimal  X[])  to
        Y[last_point]. However it is recommended to pass consistent values of Y[],
        i.e. to make Y[first_point]=Y[last_point].

          -- ALGLIB PROJECT --
             Copyright 03.09.2010 by Bochkanov Sergey
        *************************************************************************/
        public static void spline1dconvdiffcubic(double[] x,
            double[] y,
            int n,
            int boundltype,
            double boundl,
            int boundrtype,
            double boundr,
            double[] x2,
            int n2,
            ref double[] y2,
            ref double[] d2)
        {
            double[] a1 = new double[0];
            double[] a2 = new double[0];
            double[] a3 = new double[0];
            double[] b = new double[0];
            double[] d = new double[0];
            double[] dt = new double[0];
            double[] rt1 = new double[0];
            int[] p = new int[0];
            int[] p2 = new int[0];
            int i = 0;
            int ylen = 0;
            double t = 0;
            double t2 = 0;
            int i_ = 0;

            x = (double[])x.Clone();
            y = (double[])y.Clone();
            x2 = (double[])x2.Clone();
            y2 = new double[0];
            d2 = new double[0];

            
            //
            // check correctness of boundary conditions
            //
            alglib.ap.assert(((boundltype==-1 || boundltype==0) || boundltype==1) || boundltype==2, "Spline1DConvDiffCubic: incorrect BoundLType!");
            alglib.ap.assert(((boundrtype==-1 || boundrtype==0) || boundrtype==1) || boundrtype==2, "Spline1DConvDiffCubic: incorrect BoundRType!");
            alglib.ap.assert((boundrtype==-1 && boundltype==-1) || (boundrtype!=-1 && boundltype!=-1), "Spline1DConvDiffCubic: incorrect BoundLType/BoundRType!");
            if( boundltype==1 || boundltype==2 )
            {
                alglib.ap.assert(math.isfinite(boundl), "Spline1DConvDiffCubic: BoundL is infinite or NAN!");
            }
            if( boundrtype==1 || boundrtype==2 )
            {
                alglib.ap.assert(math.isfinite(boundr), "Spline1DConvDiffCubic: BoundR is infinite or NAN!");
            }
            
            //
            // check lengths of arguments
            //
            alglib.ap.assert(n>=2, "Spline1DConvDiffCubic: N<2!");
            alglib.ap.assert(alglib.ap.len(x)>=n, "Spline1DConvDiffCubic: Length(X)<N!");
            alglib.ap.assert(alglib.ap.len(y)>=n, "Spline1DConvDiffCubic: Length(Y)<N!");
            alglib.ap.assert(n2>=2, "Spline1DConvDiffCubic: N2<2!");
            alglib.ap.assert(alglib.ap.len(x2)>=n2, "Spline1DConvDiffCubic: Length(X2)<N2!");
            
            //
            // check and sort X/Y
            //
            ylen = n;
            if( boundltype==-1 )
            {
                ylen = n-1;
            }
            alglib.ap.assert(apserv.isfinitevector(x, n), "Spline1DConvDiffCubic: X contains infinite or NAN values!");
            alglib.ap.assert(apserv.isfinitevector(y, ylen), "Spline1DConvDiffCubic: Y contains infinite or NAN values!");
            alglib.ap.assert(apserv.isfinitevector(x2, n2), "Spline1DConvDiffCubic: X2 contains infinite or NAN values!");
            heapsortppoints(ref x, ref y, ref p, n);
            alglib.ap.assert(apserv.aredistinct(x, n), "Spline1DConvDiffCubic: at least two consequent points are too close!");
            
            //
            // set up DT (we will need it below)
            //
            dt = new double[Math.Max(n, n2)];
            
            //
            // sort X2:
            // * use fake array DT because HeapSortPPoints() needs both integer AND real arrays
            // * if we have periodic problem, wrap points
            // * sort them, store permutation at P2
            //
            if( boundrtype==-1 && boundltype==-1 )
            {
                for(i=0; i<=n2-1; i++)
                {
                    t = x2[i];
                    apserv.apperiodicmap(ref t, x[0], x[n-1], ref t2);
                    x2[i] = t;
                }
            }
            heapsortppoints(ref x2, ref dt, ref p2, n2);
            
            //
            // Now we've checked and preordered everything, so we:
            // * call internal GridDiff() function to get Hermite form of spline
            // * convert using internal Conv() function
            // * convert Y2 back to original order
            //
            spline1dgriddiffcubicinternal(x, ref y, n, boundltype, boundl, boundrtype, boundr, ref d, ref a1, ref a2, ref a3, ref b, ref dt);
            spline1dconvdiffinternal(x, y, d, n, x2, n2, ref y2, true, ref d2, true, ref rt1, false);
            alglib.ap.assert(alglib.ap.len(dt)>=n2, "Spline1DConvDiffCubic: internal error!");
            for(i=0; i<=n2-1; i++)
            {
                dt[p2[i]] = y2[i];
            }
            for(i_=0; i_<=n2-1;i_++)
            {
                y2[i_] = dt[i_];
            }
            for(i=0; i<=n2-1; i++)
            {
                dt[p2[i]] = d2[i];
            }
            for(i_=0; i_<=n2-1;i_++)
            {
                d2[i_] = dt[i_];
            }
        }


        /*************************************************************************
        This function solves following problem: given table y[] of function values
        at old nodes x[]  and new nodes  x2[],  it calculates and returns table of
        function  values  y2[],  first  and  second  derivatives  d2[]  and  dd2[]
        (calculated at x2[]).

        This function yields same result as Spline1DBuildCubic() call followed  by
        sequence of Spline1DDiff() calls, but it can be several times faster  when
        called for ordered X[] and X2[].

        INPUT PARAMETERS:
            X           -   old spline nodes
            Y           -   function values
            X2           -  new spline nodes

        OPTIONAL PARAMETERS:
            N           -   points count:
                            * N>=2
                            * if given, only first N points from X/Y are used
                            * if not given, automatically detected from X/Y sizes
                              (len(X) must be equal to len(Y))
            BoundLType  -   boundary condition type for the left boundary
            BoundL      -   left boundary condition (first or second derivative,
                            depending on the BoundLType)
            BoundRType  -   boundary condition type for the right boundary
            BoundR      -   right boundary condition (first or second derivative,
                            depending on the BoundRType)
            N2          -   new points count:
                            * N2>=2
                            * if given, only first N2 points from X2 are used
                            * if not given, automatically detected from X2 size

        OUTPUT PARAMETERS:
            F2          -   function values at X2[]
            D2          -   first derivatives at X2[]
            DD2         -   second derivatives at X2[]

        ORDER OF POINTS

        Subroutine automatically sorts points, so caller  may pass unsorted array.
        Function  values  are correctly reordered on  return, so F2[I]  is  always
        equal to S(X2[I]) independently of points order.

        SETTING BOUNDARY VALUES:

        The BoundLType/BoundRType parameters can have the following values:
            * -1, which corresonds to the periodic (cyclic) boundary conditions.
                  In this case:
                  * both BoundLType and BoundRType must be equal to -1.
                  * BoundL/BoundR are ignored
                  * Y[last] is ignored (it is assumed to be equal to Y[first]).
            *  0, which  corresponds  to  the  parabolically   terminated  spline
                  (BoundL and/or BoundR are ignored).
            *  1, which corresponds to the first derivative boundary condition
            *  2, which corresponds to the second derivative boundary condition
            *  by default, BoundType=0 is used

        PROBLEMS WITH PERIODIC BOUNDARY CONDITIONS:

        Problems with periodic boundary conditions have Y[first_point]=Y[last_point].
        However, this subroutine doesn't require you to specify equal  values  for
        the first and last points - it automatically forces them  to  be  equal by
        copying  Y[first_point]  (corresponds  to the leftmost,  minimal  X[])  to
        Y[last_point]. However it is recommended to pass consistent values of Y[],
        i.e. to make Y[first_point]=Y[last_point].

          -- ALGLIB PROJECT --
             Copyright 03.09.2010 by Bochkanov Sergey
        *************************************************************************/
        public static void spline1dconvdiff2cubic(double[] x,
            double[] y,
            int n,
            int boundltype,
            double boundl,
            int boundrtype,
            double boundr,
            double[] x2,
            int n2,
            ref double[] y2,
            ref double[] d2,
            ref double[] dd2)
        {
            double[] a1 = new double[0];
            double[] a2 = new double[0];
            double[] a3 = new double[0];
            double[] b = new double[0];
            double[] d = new double[0];
            double[] dt = new double[0];
            int[] p = new int[0];
            int[] p2 = new int[0];
            int i = 0;
            int ylen = 0;
            double t = 0;
            double t2 = 0;
            int i_ = 0;

            x = (double[])x.Clone();
            y = (double[])y.Clone();
            x2 = (double[])x2.Clone();
            y2 = new double[0];
            d2 = new double[0];
            dd2 = new double[0];

            
            //
            // check correctness of boundary conditions
            //
            alglib.ap.assert(((boundltype==-1 || boundltype==0) || boundltype==1) || boundltype==2, "Spline1DConvDiff2Cubic: incorrect BoundLType!");
            alglib.ap.assert(((boundrtype==-1 || boundrtype==0) || boundrtype==1) || boundrtype==2, "Spline1DConvDiff2Cubic: incorrect BoundRType!");
            alglib.ap.assert((boundrtype==-1 && boundltype==-1) || (boundrtype!=-1 && boundltype!=-1), "Spline1DConvDiff2Cubic: incorrect BoundLType/BoundRType!");
            if( boundltype==1 || boundltype==2 )
            {
                alglib.ap.assert(math.isfinite(boundl), "Spline1DConvDiff2Cubic: BoundL is infinite or NAN!");
            }
            if( boundrtype==1 || boundrtype==2 )
            {
                alglib.ap.assert(math.isfinite(boundr), "Spline1DConvDiff2Cubic: BoundR is infinite or NAN!");
            }
            
            //
            // check lengths of arguments
            //
            alglib.ap.assert(n>=2, "Spline1DConvDiff2Cubic: N<2!");
            alglib.ap.assert(alglib.ap.len(x)>=n, "Spline1DConvDiff2Cubic: Length(X)<N!");
            alglib.ap.assert(alglib.ap.len(y)>=n, "Spline1DConvDiff2Cubic: Length(Y)<N!");
            alglib.ap.assert(n2>=2, "Spline1DConvDiff2Cubic: N2<2!");
            alglib.ap.assert(alglib.ap.len(x2)>=n2, "Spline1DConvDiff2Cubic: Length(X2)<N2!");
            
            //
            // check and sort X/Y
            //
            ylen = n;
            if( boundltype==-1 )
            {
                ylen = n-1;
            }
            alglib.ap.assert(apserv.isfinitevector(x, n), "Spline1DConvDiff2Cubic: X contains infinite or NAN values!");
            alglib.ap.assert(apserv.isfinitevector(y, ylen), "Spline1DConvDiff2Cubic: Y contains infinite or NAN values!");
            alglib.ap.assert(apserv.isfinitevector(x2, n2), "Spline1DConvDiff2Cubic: X2 contains infinite or NAN values!");
            heapsortppoints(ref x, ref y, ref p, n);
            alglib.ap.assert(apserv.aredistinct(x, n), "Spline1DConvDiff2Cubic: at least two consequent points are too close!");
            
            //
            // set up DT (we will need it below)
            //
            dt = new double[Math.Max(n, n2)];
            
            //
            // sort X2:
            // * use fake array DT because HeapSortPPoints() needs both integer AND real arrays
            // * if we have periodic problem, wrap points
            // * sort them, store permutation at P2
            //
            if( boundrtype==-1 && boundltype==-1 )
            {
                for(i=0; i<=n2-1; i++)
                {
                    t = x2[i];
                    apserv.apperiodicmap(ref t, x[0], x[n-1], ref t2);
                    x2[i] = t;
                }
            }
            heapsortppoints(ref x2, ref dt, ref p2, n2);
            
            //
            // Now we've checked and preordered everything, so we:
            // * call internal GridDiff() function to get Hermite form of spline
            // * convert using internal Conv() function
            // * convert Y2 back to original order
            //
            spline1dgriddiffcubicinternal(x, ref y, n, boundltype, boundl, boundrtype, boundr, ref d, ref a1, ref a2, ref a3, ref b, ref dt);
            spline1dconvdiffinternal(x, y, d, n, x2, n2, ref y2, true, ref d2, true, ref dd2, true);
            alglib.ap.assert(alglib.ap.len(dt)>=n2, "Spline1DConvDiff2Cubic: internal error!");
            for(i=0; i<=n2-1; i++)
            {
                dt[p2[i]] = y2[i];
            }
            for(i_=0; i_<=n2-1;i_++)
            {
                y2[i_] = dt[i_];
            }
            for(i=0; i<=n2-1; i++)
            {
                dt[p2[i]] = d2[i];
            }
            for(i_=0; i_<=n2-1;i_++)
            {
                d2[i_] = dt[i_];
            }
            for(i=0; i<=n2-1; i++)
            {
                dt[p2[i]] = dd2[i];
            }
            for(i_=0; i_<=n2-1;i_++)
            {
                dd2[i_] = dt[i_];
            }
        }


        /*************************************************************************
        This subroutine builds Catmull-Rom spline interpolant.

        INPUT PARAMETERS:
            X           -   spline nodes, array[0..N-1].
            Y           -   function values, array[0..N-1].
            
        OPTIONAL PARAMETERS:
            N           -   points count:
                            * N>=2
                            * if given, only first N points are used to build spline
                            * if not given, automatically detected from X/Y sizes
                              (len(X) must be equal to len(Y))
            BoundType   -   boundary condition type:
                            * -1 for periodic boundary condition
                            *  0 for parabolically terminated spline (default)
            Tension     -   tension parameter:
                            * tension=0   corresponds to classic Catmull-Rom spline (default)
                            * 0<tension<1 corresponds to more general form - cardinal spline

        OUTPUT PARAMETERS:
            C           -   spline interpolant


        ORDER OF POINTS

        Subroutine automatically sorts points, so caller may pass unsorted array.

        PROBLEMS WITH PERIODIC BOUNDARY CONDITIONS:

        Problems with periodic boundary conditions have Y[first_point]=Y[last_point].
        However, this subroutine doesn't require you to specify equal  values  for
        the first and last points - it automatically forces them  to  be  equal by
        copying  Y[first_point]  (corresponds  to the leftmost,  minimal  X[])  to
        Y[last_point]. However it is recommended to pass consistent values of Y[],
        i.e. to make Y[first_point]=Y[last_point].

          -- ALGLIB PROJECT --
             Copyright 23.06.2007 by Bochkanov Sergey
        *************************************************************************/
        public static void spline1dbuildcatmullrom(double[] x,
            double[] y,
            int n,
            int boundtype,
            double tension,
            spline1dinterpolant c)
        {
            double[] d = new double[0];
            int i = 0;

            x = (double[])x.Clone();
            y = (double[])y.Clone();

            alglib.ap.assert(n>=2, "Spline1DBuildCatmullRom: N<2!");
            alglib.ap.assert(boundtype==-1 || boundtype==0, "Spline1DBuildCatmullRom: incorrect BoundType!");
            alglib.ap.assert((double)(tension)>=(double)(0), "Spline1DBuildCatmullRom: Tension<0!");
            alglib.ap.assert((double)(tension)<=(double)(1), "Spline1DBuildCatmullRom: Tension>1!");
            alglib.ap.assert(alglib.ap.len(x)>=n, "Spline1DBuildCatmullRom: Length(X)<N!");
            alglib.ap.assert(alglib.ap.len(y)>=n, "Spline1DBuildCatmullRom: Length(Y)<N!");
            
            //
            // check and sort points
            //
            alglib.ap.assert(apserv.isfinitevector(x, n), "Spline1DBuildCatmullRom: X contains infinite or NAN values!");
            alglib.ap.assert(apserv.isfinitevector(y, n), "Spline1DBuildCatmullRom: Y contains infinite or NAN values!");
            heapsortpoints(ref x, ref y, n);
            alglib.ap.assert(apserv.aredistinct(x, n), "Spline1DBuildCatmullRom: at least two consequent points are too close!");
            
            //
            // Special cases:
            // * N=2, parabolic terminated boundary condition on both ends
            // * N=2, periodic boundary condition
            //
            if( n==2 && boundtype==0 )
            {
                
                //
                // Just linear spline
                //
                spline1dbuildlinear(x, y, n, c);
                return;
            }
            if( n==2 && boundtype==-1 )
            {
                
                //
                // Same as cubic spline with periodic conditions
                //
                spline1dbuildcubic(x, y, n, -1, 0.0, -1, 0.0, c);
                return;
            }
            
            //
            // Periodic or non-periodic boundary conditions
            //
            if( boundtype==-1 )
            {
                
                //
                // Periodic boundary conditions
                //
                y[n-1] = y[0];
                d = new double[n];
                d[0] = (y[1]-y[n-2])/(2*(x[1]-x[0]+x[n-1]-x[n-2]));
                for(i=1; i<=n-2; i++)
                {
                    d[i] = (1-tension)*(y[i+1]-y[i-1])/(x[i+1]-x[i-1]);
                }
                d[n-1] = d[0];
                
                //
                // Now problem is reduced to the cubic Hermite spline
                //
                spline1dbuildhermite(x, y, d, n, c);
                c.periodic = true;
            }
            else
            {
                
                //
                // Non-periodic boundary conditions
                //
                d = new double[n];
                for(i=1; i<=n-2; i++)
                {
                    d[i] = (1-tension)*(y[i+1]-y[i-1])/(x[i+1]-x[i-1]);
                }
                d[0] = 2*(y[1]-y[0])/(x[1]-x[0])-d[1];
                d[n-1] = 2*(y[n-1]-y[n-2])/(x[n-1]-x[n-2])-d[n-2];
                
                //
                // Now problem is reduced to the cubic Hermite spline
                //
                spline1dbuildhermite(x, y, d, n, c);
            }
        }


        /*************************************************************************
        This subroutine builds Hermite spline interpolant.

        INPUT PARAMETERS:
            X           -   spline nodes, array[0..N-1]
            Y           -   function values, array[0..N-1]
            D           -   derivatives, array[0..N-1]
            N           -   points count (optional):
                            * N>=2
                            * if given, only first N points are used to build spline
                            * if not given, automatically detected from X/Y sizes
                              (len(X) must be equal to len(Y))

        OUTPUT PARAMETERS:
            C           -   spline interpolant.


        ORDER OF POINTS

        Subroutine automatically sorts points, so caller may pass unsorted array.

          -- ALGLIB PROJECT --
             Copyright 23.06.2007 by Bochkanov Sergey
        *************************************************************************/
        public static void spline1dbuildhermite(double[] x,
            double[] y,
            double[] d,
            int n,
            spline1dinterpolant c)
        {
            int i = 0;
            double delta = 0;
            double delta2 = 0;
            double delta3 = 0;

            x = (double[])x.Clone();
            y = (double[])y.Clone();
            d = (double[])d.Clone();

            alglib.ap.assert(n>=2, "Spline1DBuildHermite: N<2!");
            alglib.ap.assert(alglib.ap.len(x)>=n, "Spline1DBuildHermite: Length(X)<N!");
            alglib.ap.assert(alglib.ap.len(y)>=n, "Spline1DBuildHermite: Length(Y)<N!");
            alglib.ap.assert(alglib.ap.len(d)>=n, "Spline1DBuildHermite: Length(D)<N!");
            
            //
            // check and sort points
            //
            alglib.ap.assert(apserv.isfinitevector(x, n), "Spline1DBuildHermite: X contains infinite or NAN values!");
            alglib.ap.assert(apserv.isfinitevector(y, n), "Spline1DBuildHermite: Y contains infinite or NAN values!");
            alglib.ap.assert(apserv.isfinitevector(d, n), "Spline1DBuildHermite: D contains infinite or NAN values!");
            heapsortdpoints(ref x, ref y, ref d, n);
            alglib.ap.assert(apserv.aredistinct(x, n), "Spline1DBuildHermite: at least two consequent points are too close!");
            
            //
            // Build
            //
            c.x = new double[n];
            c.c = new double[4*(n-1)+2];
            c.periodic = false;
            c.k = 3;
            c.n = n;
            c.continuity = 1;
            for(i=0; i<=n-1; i++)
            {
                c.x[i] = x[i];
            }
            for(i=0; i<=n-2; i++)
            {
                delta = x[i+1]-x[i];
                delta2 = math.sqr(delta);
                delta3 = delta*delta2;
                c.c[4*i+0] = y[i];
                c.c[4*i+1] = d[i];
                c.c[4*i+2] = (3*(y[i+1]-y[i])-2*d[i]*delta-d[i+1]*delta)/delta2;
                c.c[4*i+3] = (2*(y[i]-y[i+1])+d[i]*delta+d[i+1]*delta)/delta3;
            }
            c.c[4*(n-1)+0] = y[n-1];
            c.c[4*(n-1)+1] = d[n-1];
        }


        /*************************************************************************
        This subroutine builds Akima spline interpolant

        INPUT PARAMETERS:
            X           -   spline nodes, array[0..N-1]
            Y           -   function values, array[0..N-1]
            N           -   points count (optional):
                            * N>=5
                            * if given, only first N points are used to build spline
                            * if not given, automatically detected from X/Y sizes
                              (len(X) must be equal to len(Y))

        OUTPUT PARAMETERS:
            C           -   spline interpolant


        ORDER OF POINTS

        Subroutine automatically sorts points, so caller may pass unsorted array.

          -- ALGLIB PROJECT --
             Copyright 24.06.2007 by Bochkanov Sergey
        *************************************************************************/
        public static void spline1dbuildakima(double[] x,
            double[] y,
            int n,
            spline1dinterpolant c)
        {
            int i = 0;
            double[] d = new double[0];
            double[] w = new double[0];
            double[] diff = new double[0];

            x = (double[])x.Clone();
            y = (double[])y.Clone();

            alglib.ap.assert(n>=5, "Spline1DBuildAkima: N<5!");
            alglib.ap.assert(alglib.ap.len(x)>=n, "Spline1DBuildAkima: Length(X)<N!");
            alglib.ap.assert(alglib.ap.len(y)>=n, "Spline1DBuildAkima: Length(Y)<N!");
            
            //
            // check and sort points
            //
            alglib.ap.assert(apserv.isfinitevector(x, n), "Spline1DBuildAkima: X contains infinite or NAN values!");
            alglib.ap.assert(apserv.isfinitevector(y, n), "Spline1DBuildAkima: Y contains infinite or NAN values!");
            heapsortpoints(ref x, ref y, n);
            alglib.ap.assert(apserv.aredistinct(x, n), "Spline1DBuildAkima: at least two consequent points are too close!");
            
            //
            // Prepare W (weights), Diff (divided differences)
            //
            w = new double[n-1];
            diff = new double[n-1];
            for(i=0; i<=n-2; i++)
            {
                diff[i] = (y[i+1]-y[i])/(x[i+1]-x[i]);
            }
            for(i=1; i<=n-2; i++)
            {
                w[i] = Math.Abs(diff[i]-diff[i-1]);
            }
            
            //
            // Prepare Hermite interpolation scheme
            //
            d = new double[n];
            for(i=2; i<=n-3; i++)
            {
                if( (double)(Math.Abs(w[i-1])+Math.Abs(w[i+1]))!=(double)(0) )
                {
                    d[i] = (w[i+1]*diff[i-1]+w[i-1]*diff[i])/(w[i+1]+w[i-1]);
                }
                else
                {
                    d[i] = ((x[i+1]-x[i])*diff[i-1]+(x[i]-x[i-1])*diff[i])/(x[i+1]-x[i-1]);
                }
            }
            d[0] = diffthreepoint(x[0], x[0], y[0], x[1], y[1], x[2], y[2]);
            d[1] = diffthreepoint(x[1], x[0], y[0], x[1], y[1], x[2], y[2]);
            d[n-2] = diffthreepoint(x[n-2], x[n-3], y[n-3], x[n-2], y[n-2], x[n-1], y[n-1]);
            d[n-1] = diffthreepoint(x[n-1], x[n-3], y[n-3], x[n-2], y[n-2], x[n-1], y[n-1]);
            
            //
            // Build Akima spline using Hermite interpolation scheme
            //
            spline1dbuildhermite(x, y, d, n, c);
        }


        /*************************************************************************
        This subroutine calculates the value of the spline at the given point X.

        INPUT PARAMETERS:
            C   -   spline interpolant
            X   -   point

        Result:
            S(x)

          -- ALGLIB PROJECT --
             Copyright 23.06.2007 by Bochkanov Sergey
        *************************************************************************/
        public static double spline1dcalc(spline1dinterpolant c,
            double x)
        {
            double result = 0;
            int l = 0;
            int r = 0;
            int m = 0;
            double t = 0;

            alglib.ap.assert(c.k==3, "Spline1DCalc: internal error");
            alglib.ap.assert(!Double.IsInfinity(x), "Spline1DCalc: infinite X!");
            
            //
            // special case: NaN
            //
            if( Double.IsNaN(x) )
            {
                result = Double.NaN;
                return result;
            }
            
            //
            // correct if periodic
            //
            if( c.periodic )
            {
                apserv.apperiodicmap(ref x, c.x[0], c.x[c.n-1], ref t);
            }
            
            //
            // Binary search in the [ x[0], ..., x[n-2] ] (x[n-1] is not included)
            //
            l = 0;
            r = c.n-2+1;
            while( l!=r-1 )
            {
                m = (l+r)/2;
                if( c.x[m]>=x )
                {
                    r = m;
                }
                else
                {
                    l = m;
                }
            }
            
            //
            // Interpolation
            //
            x = x-c.x[l];
            m = 4*l;
            result = c.c[m]+x*(c.c[m+1]+x*(c.c[m+2]+x*c.c[m+3]));
            return result;
        }


        /*************************************************************************
        This subroutine differentiates the spline.

        INPUT PARAMETERS:
            C   -   spline interpolant.
            X   -   point

        Result:
            S   -   S(x)
            DS  -   S'(x)
            D2S -   S''(x)

          -- ALGLIB PROJECT --
             Copyright 24.06.2007 by Bochkanov Sergey
        *************************************************************************/
        public static void spline1ddiff(spline1dinterpolant c,
            double x,
            ref double s,
            ref double ds,
            ref double d2s)
        {
            int l = 0;
            int r = 0;
            int m = 0;
            double t = 0;

            s = 0;
            ds = 0;
            d2s = 0;

            alglib.ap.assert(c.k==3, "Spline1DDiff: internal error");
            alglib.ap.assert(!Double.IsInfinity(x), "Spline1DDiff: infinite X!");
            
            //
            // special case: NaN
            //
            if( Double.IsNaN(x) )
            {
                s = Double.NaN;
                ds = Double.NaN;
                d2s = Double.NaN;
                return;
            }
            
            //
            // correct if periodic
            //
            if( c.periodic )
            {
                apserv.apperiodicmap(ref x, c.x[0], c.x[c.n-1], ref t);
            }
            
            //
            // Binary search
            //
            l = 0;
            r = c.n-2+1;
            while( l!=r-1 )
            {
                m = (l+r)/2;
                if( c.x[m]>=x )
                {
                    r = m;
                }
                else
                {
                    l = m;
                }
            }
            
            //
            // Differentiation
            //
            x = x-c.x[l];
            m = 4*l;
            s = c.c[m]+x*(c.c[m+1]+x*(c.c[m+2]+x*c.c[m+3]));
            ds = c.c[m+1]+2*x*c.c[m+2]+3*math.sqr(x)*c.c[m+3];
            d2s = 2*c.c[m+2]+6*x*c.c[m+3];
        }


        /*************************************************************************
        This subroutine makes the copy of the spline.

        INPUT PARAMETERS:
            C   -   spline interpolant.

        Result:
            CC  -   spline copy

          -- ALGLIB PROJECT --
             Copyright 29.06.2007 by Bochkanov Sergey
        *************************************************************************/
        public static void spline1dcopy(spline1dinterpolant c,
            spline1dinterpolant cc)
        {
            int s = 0;
            int i_ = 0;

            cc.periodic = c.periodic;
            cc.n = c.n;
            cc.k = c.k;
            cc.continuity = c.continuity;
            cc.x = new double[cc.n];
            for(i_=0; i_<=cc.n-1;i_++)
            {
                cc.x[i_] = c.x[i_];
            }
            s = alglib.ap.len(c.c);
            cc.c = new double[s];
            for(i_=0; i_<=s-1;i_++)
            {
                cc.c[i_] = c.c[i_];
            }
        }


        /*************************************************************************
        This subroutine unpacks the spline into the coefficients table.

        INPUT PARAMETERS:
            C   -   spline interpolant.
            X   -   point

        Result:
            Tbl -   coefficients table, unpacked format, array[0..N-2, 0..5].
                    For I = 0...N-2:
                        Tbl[I,0] = X[i]
                        Tbl[I,1] = X[i+1]
                        Tbl[I,2] = C0
                        Tbl[I,3] = C1
                        Tbl[I,4] = C2
                        Tbl[I,5] = C3
                    On [x[i], x[i+1]] spline is equals to:
                        S(x) = C0 + C1*t + C2*t^2 + C3*t^3
                        t = x-x[i]

          -- ALGLIB PROJECT --
             Copyright 29.06.2007 by Bochkanov Sergey
        *************************************************************************/
        public static void spline1dunpack(spline1dinterpolant c,
            ref int n,
            ref double[,] tbl)
        {
            int i = 0;
            int j = 0;

            n = 0;
            tbl = new double[0,0];

            tbl = new double[c.n-2+1, 2+c.k+1];
            n = c.n;
            
            //
            // Fill
            //
            for(i=0; i<=n-2; i++)
            {
                tbl[i,0] = c.x[i];
                tbl[i,1] = c.x[i+1];
                for(j=0; j<=c.k; j++)
                {
                    tbl[i,2+j] = c.c[(c.k+1)*i+j];
                }
            }
        }


        /*************************************************************************
        This subroutine performs linear transformation of the spline argument.

        INPUT PARAMETERS:
            C   -   spline interpolant.
            A, B-   transformation coefficients: x = A*t + B
        Result:
            C   -   transformed spline

          -- ALGLIB PROJECT --
             Copyright 30.06.2007 by Bochkanov Sergey
        *************************************************************************/
        public static void spline1dlintransx(spline1dinterpolant c,
            double a,
            double b)
        {
            int i = 0;
            int n = 0;
            double v = 0;
            double dv = 0;
            double d2v = 0;
            double[] x = new double[0];
            double[] y = new double[0];
            double[] d = new double[0];
            bool isperiodic = new bool();
            int contval = 0;

            alglib.ap.assert(c.k==3, "Spline1DLinTransX: internal error");
            n = c.n;
            x = new double[n];
            y = new double[n];
            d = new double[n];
            
            //
            // Unpack, X, Y, dY/dX.
            // Scale and pack with Spline1DBuildHermite again.
            //
            if( (double)(a)==(double)(0) )
            {
                
                //
                // Special case: A=0
                //
                v = spline1dcalc(c, b);
                for(i=0; i<=n-1; i++)
                {
                    x[i] = c.x[i];
                    y[i] = v;
                    d[i] = 0.0;
                }
            }
            else
            {
                
                //
                // General case, A<>0
                //
                for(i=0; i<=n-1; i++)
                {
                    x[i] = c.x[i];
                    spline1ddiff(c, x[i], ref v, ref dv, ref d2v);
                    x[i] = (x[i]-b)/a;
                    y[i] = v;
                    d[i] = a*dv;
                }
            }
            isperiodic = c.periodic;
            contval = c.continuity;
            if( contval>0 )
            {
                spline1dbuildhermite(x, y, d, n, c);
            }
            else
            {
                spline1dbuildlinear(x, y, n, c);
            }
            c.periodic = isperiodic;
            c.continuity = contval;
        }


        /*************************************************************************
        This subroutine performs linear transformation of the spline.

        INPUT PARAMETERS:
            C   -   spline interpolant.
            A, B-   transformation coefficients: S2(x) = A*S(x) + B
        Result:
            C   -   transformed spline

          -- ALGLIB PROJECT --
             Copyright 30.06.2007 by Bochkanov Sergey
        *************************************************************************/
        public static void spline1dlintransy(spline1dinterpolant c,
            double a,
            double b)
        {
            int i = 0;
            int j = 0;
            int n = 0;

            alglib.ap.assert(c.k==3, "Spline1DLinTransX: internal error");
            n = c.n;
            for(i=0; i<=n-2; i++)
            {
                c.c[4*i] = a*c.c[4*i]+b;
                for(j=1; j<=3; j++)
                {
                    c.c[4*i+j] = a*c.c[4*i+j];
                }
            }
            c.c[4*(n-1)+0] = a*c.c[4*(n-1)+0]+b;
            c.c[4*(n-1)+1] = a*c.c[4*(n-1)+1];
        }


        /*************************************************************************
        This subroutine integrates the spline.

        INPUT PARAMETERS:
            C   -   spline interpolant.
            X   -   right bound of the integration interval [a, x],
                    here 'a' denotes min(x[])
        Result:
            integral(S(t)dt,a,x)

          -- ALGLIB PROJECT --
             Copyright 23.06.2007 by Bochkanov Sergey
        *************************************************************************/
        public static double spline1dintegrate(spline1dinterpolant c,
            double x)
        {
            double result = 0;
            int n = 0;
            int i = 0;
            int j = 0;
            int l = 0;
            int r = 0;
            int m = 0;
            double w = 0;
            double v = 0;
            double t = 0;
            double intab = 0;
            double additionalterm = 0;

            n = c.n;
            
            //
            // Periodic splines require special treatment. We make
            // following transformation:
            //
            //     integral(S(t)dt,A,X) = integral(S(t)dt,A,Z)+AdditionalTerm
            //
            // here X may lie outside of [A,B], Z lies strictly in [A,B],
            // AdditionalTerm is equals to integral(S(t)dt,A,B) times some
            // integer number (may be zero).
            //
            if( c.periodic && ((double)(x)<(double)(c.x[0]) || (double)(x)>(double)(c.x[c.n-1])) )
            {
                
                //
                // compute integral(S(x)dx,A,B)
                //
                intab = 0;
                for(i=0; i<=c.n-2; i++)
                {
                    w = c.x[i+1]-c.x[i];
                    m = (c.k+1)*i;
                    intab = intab+c.c[m]*w;
                    v = w;
                    for(j=1; j<=c.k; j++)
                    {
                        v = v*w;
                        intab = intab+c.c[m+j]*v/(j+1);
                    }
                }
                
                //
                // map X into [A,B]
                //
                apserv.apperiodicmap(ref x, c.x[0], c.x[c.n-1], ref t);
                additionalterm = t*intab;
            }
            else
            {
                additionalterm = 0;
            }
            
            //
            // Binary search in the [ x[0], ..., x[n-2] ] (x[n-1] is not included)
            //
            l = 0;
            r = n-2+1;
            while( l!=r-1 )
            {
                m = (l+r)/2;
                if( (double)(c.x[m])>=(double)(x) )
                {
                    r = m;
                }
                else
                {
                    l = m;
                }
            }
            
            //
            // Integration
            //
            result = 0;
            for(i=0; i<=l-1; i++)
            {
                w = c.x[i+1]-c.x[i];
                m = (c.k+1)*i;
                result = result+c.c[m]*w;
                v = w;
                for(j=1; j<=c.k; j++)
                {
                    v = v*w;
                    result = result+c.c[m+j]*v/(j+1);
                }
            }
            w = x-c.x[l];
            m = (c.k+1)*l;
            v = w;
            result = result+c.c[m]*w;
            for(j=1; j<=c.k; j++)
            {
                v = v*w;
                result = result+c.c[m+j]*v/(j+1);
            }
            result = result+additionalterm;
            return result;
        }


        /*************************************************************************
        Internal version of Spline1DConvDiff

        Converts from Hermite spline given by grid XOld to new grid X2

        INPUT PARAMETERS:
            XOld    -   old grid
            YOld    -   values at old grid
            DOld    -   first derivative at old grid
            N       -   grid size
            X2      -   new grid
            N2      -   new grid size
            Y       -   possibly preallocated output array
                        (reallocate if too small)
            NeedY   -   do we need Y?
            D1      -   possibly preallocated output array
                        (reallocate if too small)
            NeedD1  -   do we need D1?
            D2      -   possibly preallocated output array
                        (reallocate if too small)
            NeedD2  -   do we need D1?

        OUTPUT ARRAYS:
            Y       -   values, if needed
            D1      -   first derivative, if needed
            D2      -   second derivative, if needed

          -- ALGLIB PROJECT --
             Copyright 03.09.2010 by Bochkanov Sergey
        *************************************************************************/
        public static void spline1dconvdiffinternal(double[] xold,
            double[] yold,
            double[] dold,
            int n,
            double[] x2,
            int n2,
            ref double[] y,
            bool needy,
            ref double[] d1,
            bool needd1,
            ref double[] d2,
            bool needd2)
        {
            int intervalindex = 0;
            int pointindex = 0;
            bool havetoadvance = new bool();
            double c0 = 0;
            double c1 = 0;
            double c2 = 0;
            double c3 = 0;
            double a = 0;
            double b = 0;
            double w = 0;
            double w2 = 0;
            double w3 = 0;
            double fa = 0;
            double fb = 0;
            double da = 0;
            double db = 0;
            double t = 0;

            
            //
            // Prepare space
            //
            if( needy && alglib.ap.len(y)<n2 )
            {
                y = new double[n2];
            }
            if( needd1 && alglib.ap.len(d1)<n2 )
            {
                d1 = new double[n2];
            }
            if( needd2 && alglib.ap.len(d2)<n2 )
            {
                d2 = new double[n2];
            }
            
            //
            // These assignments aren't actually needed
            // (variables are initialized in the loop below),
            // but without them compiler will complain about uninitialized locals
            //
            c0 = 0;
            c1 = 0;
            c2 = 0;
            c3 = 0;
            a = 0;
            b = 0;
            
            //
            // Cycle
            //
            intervalindex = -1;
            pointindex = 0;
            while( true )
            {
                
                //
                // are we ready to exit?
                //
                if( pointindex>=n2 )
                {
                    break;
                }
                t = x2[pointindex];
                
                //
                // do we need to advance interval?
                //
                havetoadvance = false;
                if( intervalindex==-1 )
                {
                    havetoadvance = true;
                }
                else
                {
                    if( intervalindex<n-2 )
                    {
                        havetoadvance = (double)(t)>=(double)(b);
                    }
                }
                if( havetoadvance )
                {
                    intervalindex = intervalindex+1;
                    a = xold[intervalindex];
                    b = xold[intervalindex+1];
                    w = b-a;
                    w2 = w*w;
                    w3 = w*w2;
                    fa = yold[intervalindex];
                    fb = yold[intervalindex+1];
                    da = dold[intervalindex];
                    db = dold[intervalindex+1];
                    c0 = fa;
                    c1 = da;
                    c2 = (3*(fb-fa)-2*da*w-db*w)/w2;
                    c3 = (2*(fa-fb)+da*w+db*w)/w3;
                    continue;
                }
                
                //
                // Calculate spline and its derivatives using power basis
                //
                t = t-a;
                if( needy )
                {
                    y[pointindex] = c0+t*(c1+t*(c2+t*c3));
                }
                if( needd1 )
                {
                    d1[pointindex] = c1+2*t*c2+3*t*t*c3;
                }
                if( needd2 )
                {
                    d2[pointindex] = 2*c2+6*t*c3;
                }
                pointindex = pointindex+1;
            }
        }


        /*************************************************************************
        This function finds all roots and extrema of the spline S(x)  defined  at
        [A,B] (interval which contains spline nodes).

        It  does not extrapolates function, so roots and extrema located  outside 
        of [A,B] will not be found. It returns all isolated (including  multiple)
        roots and extrema.

        INPUT PARAMETERS
            C           -   spline interpolant
            
        OUTPUT PARAMETERS
            R           -   array[NR], contains roots of the spline. 
                            In case there is no roots, this array has zero length.
            NR          -   number of roots, >=0
            DR          -   is set to True in case there is at least one interval
                            where spline is just a zero constant. Such degenerate
                            cases are not reported in the R/NR
            E           -   array[NE], contains  extrema  (maximums/minimums)  of 
                            the spline. In case there is no extrema,  this  array 
                            has zero length.
            ET          -   array[NE], extrema types:
                            * ET[i]>0 in case I-th extrema is a minimum
                            * ET[i]<0 in case I-th extrema is a maximum
            NE          -   number of extrema, >=0
            DE          -   is set to True in case there is at least one interval
                            where spline is a constant. Such degenerate cases are  
                            not reported in the E/NE.
                            
        NOTES:

        1. This function does NOT report following kinds of roots:
           * intervals where function is constantly zero
           * roots which are outside of [A,B] (note: it CAN return A or B)

        2. This function does NOT report following kinds of extrema:
           * intervals where function is a constant
           * extrema which are outside of (A,B) (note: it WON'T return A or B)
           
         -- ALGLIB PROJECT --
             Copyright 26.09.2011 by Bochkanov Sergey   
        *************************************************************************/
        public static void spline1drootsandextrema(spline1dinterpolant c,
            ref double[] r,
            ref int nr,
            ref bool dr,
            ref double[] e,
            ref int[] et,
            ref int ne,
            ref bool de)
        {
            double pl = 0;
            double ml = 0;
            double pll = 0;
            double pr = 0;
            double mr = 0;
            double[] tr = new double[0];
            double[] tmpr = new double[0];
            double[] tmpe = new double[0];
            int[] tmpet = new int[0];
            double[] tmpc = new double[0];
            double x0 = 0;
            double x1 = 0;
            double x2 = 0;
            double ex0 = 0;
            double ex1 = 0;
            int tne = 0;
            int tnr = 0;
            int i = 0;
            int j = 0;
            bool nstep = new bool();

            r = new double[0];
            nr = 0;
            dr = new bool();
            e = new double[0];
            et = new int[0];
            ne = 0;
            de = new bool();

            
            //
            //exception handling
            //
            alglib.ap.assert(c.k==3, "Spline1DRootsAndExtrema : incorrect parameter C.K!");
            alglib.ap.assert(c.continuity>=0, "Spline1DRootsAndExtrema : parameter C.Continuity must not be less than 0!");
            
            //
            //initialization of variable
            //
            nr = 0;
            ne = 0;
            dr = false;
            de = false;
            nstep = true;
            
            //
            //consider case, when C.Continuty=0
            //
            if( c.continuity==0 )
            {
                
                //
                //allocation for auxiliary arrays 
                //'TmpR ' - it stores a time value for roots
                //'TmpE ' - it stores a time value for extremums
                //'TmpET '- it stores a time value for extremums type
                //
                apserv.rvectorsetlengthatleast(ref tmpr, 3*(c.n-1));
                apserv.rvectorsetlengthatleast(ref tmpe, 2*(c.n-1));
                apserv.ivectorsetlengthatleast(ref tmpet, 2*(c.n-1));
                
                //
                //start calculating
                //
                for(i=0; i<=c.n-2; i++)
                {
                    
                    //
                    //initialization pL, mL, pR, mR
                    //
                    pl = c.c[4*i];
                    ml = c.c[4*i+1];
                    pr = c.c[4*(i+1)];
                    mr = c.c[4*i+1]+2*c.c[4*i+2]*(c.x[i+1]-c.x[i])+3*c.c[4*i+3]*(c.x[i+1]-c.x[i])*(c.x[i+1]-c.x[i]);
                    
                    //
                    //pre-searching roots and extremums
                    //
                    solvecubicpolinom(pl, ml, pr, mr, c.x[i], c.x[i+1], ref x0, ref x1, ref x2, ref ex0, ref ex1, ref tnr, ref tne, ref tr);
                    dr = dr || tnr==-1;
                    de = de || tne==-1;
                    
                    //
                    //searching of roots
                    //
                    if( tnr==1 && nstep )
                    {
                        
                        //
                        //is there roots?
                        //
                        if( nr>0 )
                        {
                            
                            //
                            //is a next root equal a previous root?
                            //if is't, then write new root
                            //
                            if( (double)(x0)!=(double)(tmpr[nr-1]) )
                            {
                                tmpr[nr] = x0;
                                nr = nr+1;
                            }
                        }
                        else
                        {
                            
                            //
                            //write a first root
                            //
                            tmpr[nr] = x0;
                            nr = nr+1;
                        }
                    }
                    else
                    {
                        
                        //
                        //case when function at a segment identically to zero
                        //then we have to clear a root, if the one located on a 
                        //constant segment
                        //
                        if( tnr==-1 )
                        {
                            
                            //
                            //safe state variable as constant
                            //
                            if( nstep )
                            {
                                nstep = false;
                            }
                            
                            //
                            //clear the root, if there is
                            //
                            if( nr>0 )
                            {
                                if( (double)(c.x[i])==(double)(tmpr[nr-1]) )
                                {
                                    nr = nr-1;
                                }
                            }
                            
                            //
                            //change state for 'DR'
                            //
                            if( !dr )
                            {
                                dr = true;
                            }
                        }
                        else
                        {
                            nstep = true;
                        }
                    }
                    
                    //
                    //searching of extremums
                    //
                    if( i>0 )
                    {
                        pll = c.c[4*(i-1)];
                        
                        //
                        //if pL=pLL or pL=pR then
                        //
                        if( tne==-1 )
                        {
                            if( !de )
                            {
                                de = true;
                            }
                        }
                        else
                        {
                            if( (double)(pl)>(double)(pll) && (double)(pl)>(double)(pr) )
                            {
                                
                                //
                                //maximum
                                //
                                tmpet[ne] = -1;
                                tmpe[ne] = c.x[i];
                                ne = ne+1;
                            }
                            else
                            {
                                if( (double)(pl)<(double)(pll) && (double)(pl)<(double)(pr) )
                                {
                                    
                                    //
                                    //minimum
                                    //
                                    tmpet[ne] = 1;
                                    tmpe[ne] = c.x[i];
                                    ne = ne+1;
                                }
                            }
                        }
                    }
                }
                
                //
                //write final result
                //
                apserv.rvectorsetlengthatleast(ref r, nr);
                apserv.rvectorsetlengthatleast(ref e, ne);
                apserv.ivectorsetlengthatleast(ref et, ne);
                
                //
                //write roots
                //
                for(i=0; i<=nr-1; i++)
                {
                    r[i] = tmpr[i];
                }
                
                //
                //write extremums and their types
                //
                for(i=0; i<=ne-1; i++)
                {
                    e[i] = tmpe[i];
                    et[i] = tmpet[i];
                }
            }
            else
            {
                
                //
                //case, when C.Continuity>=1 
                //'TmpR ' - it stores a time value for roots
                //'TmpC' - it stores a time value for extremums and 
                //their function value (TmpC={EX0,F(EX0), EX1,F(EX1), ..., EXn,F(EXn)};)
                //'TmpE' - it stores a time value for extremums only
                //'TmpET'- it stores a time value for extremums type
                //
                apserv.rvectorsetlengthatleast(ref tmpr, 2*c.n-1);
                apserv.rvectorsetlengthatleast(ref tmpc, 4*c.n);
                apserv.rvectorsetlengthatleast(ref tmpe, 2*c.n);
                apserv.ivectorsetlengthatleast(ref tmpet, 2*c.n);
                
                //
                //start calculating
                //
                for(i=0; i<=c.n-2; i++)
                {
                    
                    //
                    //we calculate pL,mL, pR,mR as Fi+1(F'i+1) at left border
                    //
                    pl = c.c[4*i];
                    ml = c.c[4*i+1];
                    pr = c.c[4*(i+1)];
                    mr = c.c[4*(i+1)+1];
                    
                    //
                    //calculating roots and extremums at [X[i],X[i+1]]
                    //
                    solvecubicpolinom(pl, ml, pr, mr, c.x[i], c.x[i+1], ref x0, ref x1, ref x2, ref ex0, ref ex1, ref tnr, ref tne, ref tr);
                    
                    //
                    //searching roots
                    //
                    if( tnr>0 )
                    {
                        
                        //
                        //re-init tR
                        //
                        if( tnr>=1 )
                        {
                            tr[0] = x0;
                        }
                        if( tnr>=2 )
                        {
                            tr[1] = x1;
                        }
                        if( tnr==3 )
                        {
                            tr[2] = x2;
                        }
                        
                        //
                        //start root selection
                        //
                        if( nr>0 )
                        {
                            if( (double)(tmpr[nr-1])!=(double)(x0) )
                            {
                                
                                //
                                //previous segment was't constant identical zero
                                //
                                if( nstep )
                                {
                                    for(j=0; j<=tnr-1; j++)
                                    {
                                        tmpr[nr+j] = tr[j];
                                    }
                                    nr = nr+tnr;
                                }
                                else
                                {
                                    
                                    //
                                    //previous segment was constant identical zero
                                    //and we must ignore [NR+j-1] root
                                    //
                                    for(j=1; j<=tnr-1; j++)
                                    {
                                        tmpr[nr+j-1] = tr[j];
                                    }
                                    nr = nr+tnr-1;
                                    nstep = true;
                                }
                            }
                            else
                            {
                                for(j=1; j<=tnr-1; j++)
                                {
                                    tmpr[nr+j-1] = tr[j];
                                }
                                nr = nr+tnr-1;
                            }
                        }
                        else
                        {
                            
                            //
                            //write first root
                            //
                            for(j=0; j<=tnr-1; j++)
                            {
                                tmpr[nr+j] = tr[j];
                            }
                            nr = nr+tnr;
                        }
                    }
                    else
                    {
                        if( tnr==-1 )
                        {
                            
                            //
                            //decrement 'NR' if at previous step was writen a root
                            //(previous segment identical zero)
                            //
                            if( nr>0 && nstep )
                            {
                                nr = nr-1;
                            }
                            
                            //
                            //previous segment is't constant
                            //
                            if( nstep )
                            {
                                nstep = false;
                            }
                            
                            //
                            //rewrite 'DR'
                            //
                            if( !dr )
                            {
                                dr = true;
                            }
                        }
                    }
                    
                    //
                    //searching extremums
                    //write all term like extremums
                    //
                    if( tne==1 )
                    {
                        if( ne>0 )
                        {
                            
                            //
                            //just ignore identical extremums
                            //because he must be one
                            //
                            if( (double)(tmpc[ne-2])!=(double)(ex0) )
                            {
                                tmpc[ne] = ex0;
                                tmpc[ne+1] = c.c[4*i]+c.c[4*i+1]*(ex0-c.x[i])+c.c[4*i+2]*(ex0-c.x[i])*(ex0-c.x[i])+c.c[4*i+3]*(ex0-c.x[i])*(ex0-c.x[i])*(ex0-c.x[i]);
                                ne = ne+2;
                            }
                        }
                        else
                        {
                            
                            //
                            //write first extremum and it function value
                            //
                            tmpc[ne] = ex0;
                            tmpc[ne+1] = c.c[4*i]+c.c[4*i+1]*(ex0-c.x[i])+c.c[4*i+2]*(ex0-c.x[i])*(ex0-c.x[i])+c.c[4*i+3]*(ex0-c.x[i])*(ex0-c.x[i])*(ex0-c.x[i]);
                            ne = ne+2;
                        }
                    }
                    else
                    {
                        if( tne==2 )
                        {
                            if( ne>0 )
                            {
                                
                                //
                                //ignore identical extremum
                                //
                                if( (double)(tmpc[ne-2])!=(double)(ex0) )
                                {
                                    tmpc[ne] = ex0;
                                    tmpc[ne+1] = c.c[4*i]+c.c[4*i+1]*(ex0-c.x[i])+c.c[4*i+2]*(ex0-c.x[i])*(ex0-c.x[i])+c.c[4*i+3]*(ex0-c.x[i])*(ex0-c.x[i])*(ex0-c.x[i]);
                                    ne = ne+2;
                                }
                            }
                            else
                            {
                                
                                //
                                //write first extremum
                                //
                                tmpc[ne] = ex0;
                                tmpc[ne+1] = c.c[4*i]+c.c[4*i+1]*(ex0-c.x[i])+c.c[4*i+2]*(ex0-c.x[i])*(ex0-c.x[i])+c.c[4*i+3]*(ex0-c.x[i])*(ex0-c.x[i])*(ex0-c.x[i]);
                                ne = ne+2;
                            }
                            
                            //
                            //write second extremum
                            //
                            tmpc[ne] = ex1;
                            tmpc[ne+1] = c.c[4*i]+c.c[4*i+1]*(ex1-c.x[i])+c.c[4*i+2]*(ex1-c.x[i])*(ex1-c.x[i])+c.c[4*i+3]*(ex1-c.x[i])*(ex1-c.x[i])*(ex1-c.x[i]);
                            ne = ne+2;
                        }
                        else
                        {
                            if( tne==-1 )
                            {
                                if( !de )
                                {
                                    de = true;
                                }
                            }
                        }
                    }
                }
                
                //
                //checking of arrays
                //get number of extremums (tNe=NE/2)
                //initialize pL as value F0(X[0]) and
                //initialize pR as value Fn-1(X[N])
                //
                tne = ne/2;
                ne = 0;
                pl = c.c[0];
                pr = c.c[4*(c.n-1)];
                for(i=0; i<=tne-1; i++)
                {
                    if( i>0 && i<tne-1 )
                    {
                        if( (double)(tmpc[2*i+1])>(double)(tmpc[2*(i-1)+1]) && (double)(tmpc[2*i+1])>(double)(tmpc[2*(i+1)+1]) )
                        {
                            
                            //
                            //maximum
                            //
                            tmpe[ne] = tmpc[2*i];
                            tmpet[ne] = -1;
                            ne = ne+1;
                        }
                        else
                        {
                            if( (double)(tmpc[2*i+1])<(double)(tmpc[2*(i-1)+1]) && (double)(tmpc[2*i+1])<(double)(tmpc[2*(i+1)+1]) )
                            {
                                
                                //
                                //minimum
                                //
                                tmpe[ne] = tmpc[2*i];
                                tmpet[ne] = 1;
                                ne = ne+1;
                            }
                        }
                    }
                    else
                    {
                        if( i==0 )
                        {
                            if( (double)(tmpc[2*i])!=(double)(c.x[0]) )
                            {
                                if( (double)(tmpc[2*i+1])>(double)(pl) && (double)(tmpc[2*i+1])>(double)(tmpc[2*(i+1)+1]) )
                                {
                                    
                                    //
                                    //maximum
                                    //
                                    tmpe[ne] = tmpc[2*i];
                                    tmpet[ne] = -1;
                                    ne = ne+1;
                                }
                                else
                                {
                                    if( (double)(tmpc[2*i+1])<(double)(pl) && (double)(tmpc[2*i+1])<(double)(tmpc[2*(i+1)+1]) )
                                    {
                                        
                                        //
                                        //minimum
                                        //
                                        tmpe[ne] = tmpc[2*i];
                                        tmpet[ne] = 1;
                                        ne = ne+1;
                                    }
                                }
                            }
                        }
                        else
                        {
                            if( i==tne-1 )
                            {
                                if( (double)(tmpc[2*i])!=(double)(c.x[c.n-1]) )
                                {
                                    if( (double)(tmpc[2*i+1])>(double)(tmpc[2*(i-1)+1]) && (double)(tmpc[2*i+1])>(double)(pr) )
                                    {
                                        
                                        //
                                        //maximum
                                        //
                                        tmpe[ne] = tmpc[2*i];
                                        tmpet[ne] = -1;
                                        ne = ne+1;
                                    }
                                    else
                                    {
                                        if( (double)(tmpc[2*i+1])<(double)(tmpc[2*(i-1)+1]) && (double)(tmpc[2*i+1])<(double)(pr) )
                                        {
                                            
                                            //
                                            //minimum
                                            //
                                            tmpe[ne] = tmpc[2*i];
                                            tmpet[ne] = 1;
                                            ne = ne+1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                
                //
                //final results
                //allocate R, E, ET
                //
                apserv.rvectorsetlengthatleast(ref r, nr);
                apserv.rvectorsetlengthatleast(ref e, ne);
                apserv.ivectorsetlengthatleast(ref et, ne);
                
                //
                //write result for extremus and their types
                //
                for(i=0; i<=ne-1; i++)
                {
                    e[i] = tmpe[i];
                    et[i] = tmpet[i];
                }
                
                //
                //write result for roots
                //
                for(i=0; i<=nr-1; i++)
                {
                    r[i] = tmpr[i];
                }
            }
        }


        /*************************************************************************
        Internal subroutine. Heap sort.
        *************************************************************************/
        public static void heapsortdpoints(ref double[] x,
            ref double[] y,
            ref double[] d,
            int n)
        {
            double[] rbuf = new double[0];
            int[] ibuf = new int[0];
            double[] rbuf2 = new double[0];
            int[] ibuf2 = new int[0];
            int i = 0;
            int i_ = 0;

            ibuf = new int[n];
            rbuf = new double[n];
            for(i=0; i<=n-1; i++)
            {
                ibuf[i] = i;
            }
            tsort.tagsortfasti(ref x, ref ibuf, ref rbuf2, ref ibuf2, n);
            for(i=0; i<=n-1; i++)
            {
                rbuf[i] = y[ibuf[i]];
            }
            for(i_=0; i_<=n-1;i_++)
            {
                y[i_] = rbuf[i_];
            }
            for(i=0; i<=n-1; i++)
            {
                rbuf[i] = d[ibuf[i]];
            }
            for(i_=0; i_<=n-1;i_++)
            {
                d[i_] = rbuf[i_];
            }
        }


        /*************************************************************************
        This procedure search roots of an quadratic equation inside [0;1] and it number of roots.

        INPUT PARAMETERS:
            P0   -   value of a function at 0
            M0   -   value of a derivative at 0
            P1   -   value of a function at 1
            M1   -   value of a derivative at 1

        OUTPUT PARAMETERS:
            X0   -  first root of an equation
            X1   -  second root of an equation
            NR   -  number of roots
            
        RESTRICTIONS OF PARAMETERS:

        Parameters for this procedure has't to be zero simultaneously. Is expected, 
        that input polinom is't degenerate or constant identicaly ZERO.


        REMARK:

        The procedure always fill value for X1 and X2, even if it is't belongs to [0;1].
        But first true root(even if existing one) is in X1.
        Number of roots is NR.

         -- ALGLIB PROJECT --
             Copyright 26.09.2011 by Bochkanov Sergey
        *************************************************************************/
        public static void solvepolinom2(double p0,
            double m0,
            double p1,
            double m1,
            ref double x0,
            ref double x1,
            ref int nr)
        {
            double a = 0;
            double b = 0;
            double c = 0;
            double dd = 0;
            double tmp = 0;
            double exf = 0;
            double extr = 0;

            x0 = 0;
            x1 = 0;
            nr = 0;

            
            //
            //calculate parameters for equation: A, B  and C
            //
            a = 6*p0+3*m0-6*p1+3*m1;
            b = -(6*p0)-4*m0+6*p1-2*m1;
            c = m0;
            
            //
            //check case, when A=0
            //we are considering the linear equation
            //
            if( (double)(a)==(double)(0) )
            {
                
                //
                //B<>0 and root inside [0;1]
                //one root
                //
                if( ((double)(b)!=(double)(0) && Math.Sign(c)*Math.Sign(b)<=0) && (double)(Math.Abs(b))>=(double)(Math.Abs(c)) )
                {
                    x0 = -(c/b);
                    nr = 1;
                    return;
                }
                else
                {
                    nr = 0;
                    return;
                }
            }
            
            //
            //consider case, when extremumu outside (0;1)
            //exist one root only
            //
            if( (double)(Math.Abs(2*a))<=(double)(Math.Abs(b)) || Math.Sign(b)*Math.Sign(a)>=0 )
            {
                if( Math.Sign(m0)*Math.Sign(m1)>0 )
                {
                    nr = 0;
                    return;
                }
                
                //
                //consider case, when the one exist
                //same sign of derivative
                //
                if( Math.Sign(m0)*Math.Sign(m1)<0 )
                {
                    nr = 1;
                    extr = -(b/(2*a));
                    dd = b*b-4*a*c;
                    if( (double)(dd)<(double)(0) )
                    {
                        return;
                    }
                    x0 = (-b-Math.Sqrt(dd))/(2*a);
                    x1 = (-b+Math.Sqrt(dd))/(2*a);
                    if( ((double)(extr)>=(double)(1) && (double)(x1)<=(double)(extr)) || ((double)(extr)<=(double)(0) && (double)(x1)>=(double)(extr)) )
                    {
                        x0 = x1;
                    }
                    return;
                }
                
                //
                //consider case, when the one is 0
                //
                if( (double)(m0)==(double)(0) )
                {
                    x0 = 0;
                    nr = 1;
                    return;
                }
                if( (double)(m1)==(double)(0) )
                {
                    x0 = 1;
                    nr = 1;
                    return;
                }
            }
            else
            {
                
                //
                //consider case, when both of derivatives is 0
                //
                if( (double)(m0)==(double)(0) && (double)(m1)==(double)(0) )
                {
                    x0 = 0;
                    x1 = 1;
                    nr = 2;
                    return;
                }
                
                //
                //consider case, when derivative at 0 is 0, and derivative at 1 is't 0
                //
                if( (double)(m0)==(double)(0) && (double)(m1)!=(double)(0) )
                {
                    dd = b*b-4*a*c;
                    if( (double)(dd)<(double)(0) )
                    {
                        x0 = 0;
                        nr = 1;
                        return;
                    }
                    x0 = (-b-Math.Sqrt(dd))/(2*a);
                    x1 = (-b+Math.Sqrt(dd))/(2*a);
                    extr = -(b/(2*a));
                    exf = a*extr*extr+b*extr+c;
                    if( Math.Sign(exf)*Math.Sign(m1)>0 )
                    {
                        x0 = 0;
                        nr = 1;
                        return;
                    }
                    else
                    {
                        if( (double)(extr)>(double)(x0) )
                        {
                            x0 = 0;
                        }
                        else
                        {
                            x1 = 0;
                        }
                        nr = 2;
                        
                        //
                        //roots must placed ascending
                        //
                        if( (double)(x0)>(double)(x1) )
                        {
                            tmp = x0;
                            x0 = x1;
                            x1 = tmp;
                        }
                        return;
                    }
                }
                if( (double)(m1)==(double)(0) && (double)(m0)!=(double)(0) )
                {
                    dd = b*b-4*a*c;
                    if( (double)(dd)<(double)(0) )
                    {
                        x0 = 1;
                        nr = 1;
                        return;
                    }
                    x0 = (-b-Math.Sqrt(dd))/(2*a);
                    x1 = (-b+Math.Sqrt(dd))/(2*a);
                    extr = -(b/(2*a));
                    exf = a*extr*extr+b*extr+c;
                    if( Math.Sign(exf)*Math.Sign(m0)>0 )
                    {
                        x0 = 1;
                        nr = 1;
                        return;
                    }
                    else
                    {
                        if( (double)(extr)<(double)(x0) )
                        {
                            x0 = 1;
                        }
                        else
                        {
                            x1 = 1;
                        }
                        nr = 2;
                        
                        //
                        //roots must placed ascending
                        //
                        if( (double)(x0)>(double)(x1) )
                        {
                            tmp = x0;
                            x0 = x1;
                            x1 = tmp;
                        }
                        return;
                    }
                }
                else
                {
                    extr = -(b/(2*a));
                    exf = a*extr*extr+b*extr+c;
                    if( Math.Sign(exf)*Math.Sign(m0)>0 && Math.Sign(exf)*Math.Sign(m1)>0 )
                    {
                        nr = 0;
                        return;
                    }
                    dd = b*b-4*a*c;
                    if( (double)(dd)<(double)(0) )
                    {
                        nr = 0;
                        return;
                    }
                    x0 = (-b-Math.Sqrt(dd))/(2*a);
                    x1 = (-b+Math.Sqrt(dd))/(2*a);
                    
                    //
                    //if EXF and m0, EXF and m1 has different signs, then equation has two roots              
                    //
                    if( Math.Sign(exf)*Math.Sign(m0)<0 && Math.Sign(exf)*Math.Sign(m1)<0 )
                    {
                        nr = 2;
                        
                        //
                        //roots must placed ascending
                        //
                        if( (double)(x0)>(double)(x1) )
                        {
                            tmp = x0;
                            x0 = x1;
                            x1 = tmp;
                        }
                        return;
                    }
                    else
                    {
                        nr = 1;
                        if( Math.Sign(exf)*Math.Sign(m0)<0 )
                        {
                            if( (double)(x1)<(double)(extr) )
                            {
                                x0 = x1;
                            }
                            return;
                        }
                        if( Math.Sign(exf)*Math.Sign(m1)<0 )
                        {
                            if( (double)(x1)>(double)(extr) )
                            {
                                x0 = x1;
                            }
                            return;
                        }
                    }
                }
            }
        }


        /*************************************************************************
        This procedure search roots of an cubic equation inside [A;B], it number of roots 
        and number of extremums.

        INPUT PARAMETERS:
            pA   -   value of a function at A
            mA   -   value of a derivative at A
            pB   -   value of a function at B
            mB   -   value of a derivative at B
            A0   -   left border [A0;B0]
            B0   -   right border [A0;B0]

        OUTPUT PARAMETERS:
            X0   -  first root of an equation
            X1   -  second root of an equation
            X2   -  third root of an equation
            EX0  -  first extremum of a function
            EX0  -  second extremum of a function
            NR   -  number of roots
            NR   -  number of extrmums
            
        RESTRICTIONS OF PARAMETERS:

        Length of [A;B] must be positive and is't zero, i.e. A<>B and A<B.


        REMARK:

        If 'NR' is -1 it's mean, than polinom has infiniti roots.
        If 'NE' is -1 it's mean, than polinom has infiniti extremums.

         -- ALGLIB PROJECT --
             Copyright 26.09.2011 by Bochkanov Sergey
        *************************************************************************/
        public static void solvecubicpolinom(double pa,
            double ma,
            double pb,
            double mb,
            double a,
            double b,
            ref double x0,
            ref double x1,
            ref double x2,
            ref double ex0,
            ref double ex1,
            ref int nr,
            ref int ne,
            ref double[] tempdata)
        {
            int i = 0;
            double tmpma = 0;
            double tmpmb = 0;
            double tex0 = 0;
            double tex1 = 0;

            x0 = 0;
            x1 = 0;
            x2 = 0;
            ex0 = 0;
            ex1 = 0;
            nr = 0;
            ne = 0;

            apserv.rvectorsetlengthatleast(ref tempdata, 3);
            
            //
            //case, when A>B
            //
            alglib.ap.assert((double)(a)<(double)(b), "\nSolveCubicPolinom: incorrect borders for [A;B]!\n");
            
            //
            //case 1    
            //function can be identicaly to ZERO
            //
            if( (((double)(ma)==(double)(0) && (double)(mb)==(double)(0)) && (double)(pa)==(double)(pb)) && (double)(pa)==(double)(0) )
            {
                nr = -1;
                ne = -1;
                return;
            }
            if( ((double)(ma)==(double)(0) && (double)(mb)==(double)(0)) && (double)(pa)==(double)(pb) )
            {
                nr = 0;
                ne = -1;
                return;
            }
            tmpma = ma*(b-a);
            tmpmb = mb*(b-a);
            solvepolinom2(pa, tmpma, pb, tmpmb, ref ex0, ref ex1, ref ne);
            ex0 = rescaleval(0, 1, a, b, ex0);
            ex1 = rescaleval(0, 1, a, b, ex1);
            
            //
            //case 3.1
            //no extremums at [A;B]
            //
            if( ne==0 )
            {
                nr = bisectmethod(pa, tmpma, pb, tmpmb, 0, 1, ref x0);
                if( nr==1 )
                {
                    x0 = rescaleval(0, 1, a, b, x0);
                }
                return;
            }
            
            //
            //case 3.2
            //one extremum
            //
            if( ne==1 )
            {
                if( (double)(ex0)==(double)(a) || (double)(ex0)==(double)(b) )
                {
                    nr = bisectmethod(pa, tmpma, pb, tmpmb, 0, 1, ref x0);
                    if( nr==1 )
                    {
                        x0 = rescaleval(0, 1, a, b, x0);
                    }
                    return;
                }
                else
                {
                    nr = 0;
                    i = 0;
                    tex0 = rescaleval(a, b, 0, 1, ex0);
                    nr = bisectmethod(pa, tmpma, pb, tmpmb, 0, tex0, ref x0)+nr;
                    if( nr>i )
                    {
                        tempdata[i] = rescaleval(0, tex0, a, ex0, x0);
                        i = i+1;
                    }
                    nr = bisectmethod(pa, tmpma, pb, tmpmb, tex0, 1, ref x0)+nr;
                    if( nr>i )
                    {
                        x0 = rescaleval(tex0, 1, ex0, b, x0);
                        if( i>0 )
                        {
                            if( (double)(x0)!=(double)(tempdata[i-1]) )
                            {
                                tempdata[i] = x0;
                                i = i+1;
                            }
                            else
                            {
                                nr = nr-1;
                            }
                        }
                        else
                        {
                            tempdata[i] = x0;
                            i = i+1;
                        }
                    }
                    if( nr>0 )
                    {
                        x0 = tempdata[0];
                        if( nr>1 )
                        {
                            x1 = tempdata[1];
                        }
                        return;
                    }
                }
                return;
            }
            else
            {
                
                //
                //case 3.3
                //two extremums(or more, but it's impossible)
                //
                //
                //case 3.3.0
                //both extremums at the border
                //
                if( (double)(ex0)==(double)(a) && (double)(ex1)==(double)(b) )
                {
                    nr = bisectmethod(pa, tmpma, pb, tmpmb, 0, 1, ref x0);
                    if( nr==1 )
                    {
                        x0 = rescaleval(0, 1, a, b, x0);
                    }
                    return;
                }
                if( (double)(ex0)==(double)(a) && (double)(ex1)!=(double)(b) )
                {
                    nr = 0;
                    i = 0;
                    tex1 = rescaleval(a, b, 0, 1, ex1);
                    nr = bisectmethod(pa, tmpma, pb, tmpmb, 0, tex1, ref x0)+nr;
                    if( nr>i )
                    {
                        tempdata[i] = rescaleval(0, tex1, a, ex1, x0);
                        i = i+1;
                    }
                    nr = bisectmethod(pa, tmpma, pb, tmpmb, tex1, 1, ref x0)+nr;
                    if( nr>i )
                    {
                        x0 = rescaleval(tex1, 1, ex1, b, x0);
                        if( (double)(x0)!=(double)(tempdata[i-1]) )
                        {
                            tempdata[i] = x0;
                            i = i+1;
                        }
                        else
                        {
                            nr = nr-1;
                        }
                    }
                    if( nr>0 )
                    {
                        x0 = tempdata[0];
                        if( nr>1 )
                        {
                            x1 = tempdata[1];
                        }
                        return;
                    }
                }
                if( (double)(ex1)==(double)(b) && (double)(ex0)!=(double)(a) )
                {
                    nr = 0;
                    i = 0;
                    tex0 = rescaleval(a, b, 0, 1, ex0);
                    nr = bisectmethod(pa, tmpma, pb, tmpmb, 0, tex0, ref x0)+nr;
                    if( nr>i )
                    {
                        tempdata[i] = rescaleval(0, tex0, a, ex0, x0);
                        i = i+1;
                    }
                    nr = bisectmethod(pa, tmpma, pb, tmpmb, tex0, 1, ref x0)+nr;
                    if( nr>i )
                    {
                        x0 = rescaleval(tex0, 1, ex0, b, x0);
                        if( i>0 )
                        {
                            if( (double)(x0)!=(double)(tempdata[i-1]) )
                            {
                                tempdata[i] = x0;
                                i = i+1;
                            }
                            else
                            {
                                nr = nr-1;
                            }
                        }
                        else
                        {
                            tempdata[i] = x0;
                            i = i+1;
                        }
                    }
                    if( nr>0 )
                    {
                        x0 = tempdata[0];
                        if( nr>1 )
                        {
                            x1 = tempdata[1];
                        }
                        return;
                    }
                }
                else
                {
                    
                    //
                    //case 3.3.2
                    //both extremums inside (0;1)
                    //
                    nr = 0;
                    i = 0;
                    tex0 = rescaleval(a, b, 0, 1, ex0);
                    tex1 = rescaleval(a, b, 0, 1, ex1);
                    nr = bisectmethod(pa, tmpma, pb, tmpmb, 0, tex0, ref x0)+nr;
                    if( nr>i )
                    {
                        tempdata[i] = rescaleval(0, tex0, a, ex0, x0);
                        i = i+1;
                    }
                    nr = bisectmethod(pa, tmpma, pb, tmpmb, tex0, tex1, ref x0)+nr;
                    if( nr>i )
                    {
                        x0 = rescaleval(tex0, tex1, ex0, ex1, x0);
                        if( i>0 )
                        {
                            if( (double)(x0)!=(double)(tempdata[i-1]) )
                            {
                                tempdata[i] = x0;
                                i = i+1;
                            }
                            else
                            {
                                nr = nr-1;
                            }
                        }
                        else
                        {
                            tempdata[i] = x0;
                            i = i+1;
                        }
                    }
                    nr = bisectmethod(pa, tmpma, pb, tmpmb, tex1, 1, ref x0)+nr;
                    if( nr>i )
                    {
                        x0 = rescaleval(tex1, 1, ex1, b, x0);
                        if( i>0 )
                        {
                            if( (double)(x0)!=(double)(tempdata[i-1]) )
                            {
                                tempdata[i] = x0;
                                i = i+1;
                            }
                            else
                            {
                                nr = nr-1;
                            }
                        }
                        else
                        {
                            tempdata[i] = x0;
                            i = i+1;
                        }
                    }
                    
                    //
                    //write are found roots
                    //
                    if( nr>0 )
                    {
                        x0 = tempdata[0];
                        if( nr>1 )
                        {
                            x1 = tempdata[1];
                        }
                        if( nr>2 )
                        {
                            x2 = tempdata[2];
                        }
                        return;
                    }
                }
            }
        }


        /*************************************************************************
        Function for searching a root at [A;B] by bisection method and return number of roots
        (0 or 1)

        INPUT PARAMETERS:
            pA   -   value of a function at A
            mA   -   value of a derivative at A
            pB   -   value of a function at B
            mB   -   value of a derivative at B
            A0   -   left border [A0;B0] 
            B0   -   right border [A0;B0]
            
        RESTRICTIONS OF PARAMETERS:

        We assume, that B0>A0.


        REMARK:

        Assume, that exist one root only at [A;B], else 
        function may be work incorrectly.
        The function dont check value A0,B0!

         -- ALGLIB PROJECT --
             Copyright 26.09.2011 by Bochkanov Sergey
        *************************************************************************/
        public static int bisectmethod(double pa,
            double ma,
            double pb,
            double mb,
            double a,
            double b,
            ref double x)
        {
            int result = 0;
            double vacuum = 0;
            double eps = 0;
            double a0 = 0;
            double b0 = 0;
            double m = 0;
            double lf = 0;
            double rf = 0;
            double mf = 0;

            x = 0;

            
            //
            //accuracy
            //
            eps = 1000*(b-a)*math.machineepsilon;
            
            //
            //initialization left and right borders
            //
            a0 = a;
            b0 = b;
            
            //
            //initialize function value at 'A' and 'B'
            //
            hermitecalc(pa, ma, pb, mb, a, ref lf, ref vacuum);
            hermitecalc(pa, ma, pb, mb, b, ref rf, ref vacuum);
            
            //
            //check, that 'A' and 'B' are't roots,
            //and that root exist
            //
            if( Math.Sign(lf)*Math.Sign(rf)>0 )
            {
                result = 0;
                return result;
            }
            else
            {
                if( (double)(lf)==(double)(0) )
                {
                    x = a;
                    result = 1;
                    return result;
                }
                else
                {
                    if( (double)(rf)==(double)(0) )
                    {
                        x = b;
                        result = 1;
                        return result;
                    }
                }
            }
            
            //
            //searching a root
            //
            do
            {
                m = (b0+a0)/2;
                hermitecalc(pa, ma, pb, mb, a0, ref lf, ref vacuum);
                hermitecalc(pa, ma, pb, mb, b0, ref rf, ref vacuum);
                hermitecalc(pa, ma, pb, mb, m, ref mf, ref vacuum);
                if( Math.Sign(mf)*Math.Sign(lf)<0 )
                {
                    b0 = m;
                }
                else
                {
                    if( Math.Sign(mf)*Math.Sign(rf)<0 )
                    {
                        a0 = m;
                    }
                    else
                    {
                        if( (double)(lf)==(double)(0) )
                        {
                            x = a0;
                            result = 1;
                            return result;
                        }
                        if( (double)(rf)==(double)(0) )
                        {
                            x = b0;
                            result = 1;
                            return result;
                        }
                        if( (double)(mf)==(double)(0) )
                        {
                            x = m;
                            result = 1;
                            return result;
                        }
                    }
                }
            }
            while( (double)(Math.Abs(b0-a0))>=(double)(eps) );
            x = m;
            result = 1;
            return result;
        }


        /*************************************************************************
        Internal version of Spline1DGridDiffCubic.

        Accepts pre-ordered X/Y, temporary arrays (which may be  preallocated,  if
        you want to save time, or not) and output array (which may be preallocated
        too).

        Y is passed as var-parameter because we may need to force last element  to
        be equal to the first one (if periodic boundary conditions are specified).

          -- ALGLIB PROJECT --
             Copyright 03.09.2010 by Bochkanov Sergey
        *************************************************************************/
        private static void spline1dgriddiffcubicinternal(double[] x,
            ref double[] y,
            int n,
            int boundltype,
            double boundl,
            int boundrtype,
            double boundr,
            ref double[] d,
            ref double[] a1,
            ref double[] a2,
            ref double[] a3,
            ref double[] b,
            ref double[] dt)
        {
            int i = 0;
            int i_ = 0;

            
            //
            // allocate arrays
            //
            if( alglib.ap.len(d)<n )
            {
                d = new double[n];
            }
            if( alglib.ap.len(a1)<n )
            {
                a1 = new double[n];
            }
            if( alglib.ap.len(a2)<n )
            {
                a2 = new double[n];
            }
            if( alglib.ap.len(a3)<n )
            {
                a3 = new double[n];
            }
            if( alglib.ap.len(b)<n )
            {
                b = new double[n];
            }
            if( alglib.ap.len(dt)<n )
            {
                dt = new double[n];
            }
            
            //
            // Special cases:
            // * N=2, parabolic terminated boundary condition on both ends
            // * N=2, periodic boundary condition
            //
            if( (n==2 && boundltype==0) && boundrtype==0 )
            {
                d[0] = (y[1]-y[0])/(x[1]-x[0]);
                d[1] = d[0];
                return;
            }
            if( (n==2 && boundltype==-1) && boundrtype==-1 )
            {
                d[0] = 0;
                d[1] = 0;
                return;
            }
            
            //
            // Periodic and non-periodic boundary conditions are
            // two separate classes
            //
            if( boundrtype==-1 && boundltype==-1 )
            {
                
                //
                // Periodic boundary conditions
                //
                y[n-1] = y[0];
                
                //
                // Boundary conditions at N-1 points
                // (one point less because last point is the same as first point).
                //
                a1[0] = x[1]-x[0];
                a2[0] = 2*(x[1]-x[0]+x[n-1]-x[n-2]);
                a3[0] = x[n-1]-x[n-2];
                b[0] = 3*(y[n-1]-y[n-2])/(x[n-1]-x[n-2])*(x[1]-x[0])+3*(y[1]-y[0])/(x[1]-x[0])*(x[n-1]-x[n-2]);
                for(i=1; i<=n-2; i++)
                {
                    
                    //
                    // Altough last point is [N-2], we use X[N-1] and Y[N-1]
                    // (because of periodicity)
                    //
                    a1[i] = x[i+1]-x[i];
                    a2[i] = 2*(x[i+1]-x[i-1]);
                    a3[i] = x[i]-x[i-1];
                    b[i] = 3*(y[i]-y[i-1])/(x[i]-x[i-1])*(x[i+1]-x[i])+3*(y[i+1]-y[i])/(x[i+1]-x[i])*(x[i]-x[i-1]);
                }
                
                //
                // Solve, add last point (with index N-1)
                //
                solvecyclictridiagonal(a1, a2, a3, b, n-1, ref dt);
                for(i_=0; i_<=n-2;i_++)
                {
                    d[i_] = dt[i_];
                }
                d[n-1] = d[0];
            }
            else
            {
                
                //
                // Non-periodic boundary condition.
                // Left boundary conditions.
                //
                if( boundltype==0 )
                {
                    a1[0] = 0;
                    a2[0] = 1;
                    a3[0] = 1;
                    b[0] = 2*(y[1]-y[0])/(x[1]-x[0]);
                }
                if( boundltype==1 )
                {
                    a1[0] = 0;
                    a2[0] = 1;
                    a3[0] = 0;
                    b[0] = boundl;
                }
                if( boundltype==2 )
                {
                    a1[0] = 0;
                    a2[0] = 2;
                    a3[0] = 1;
                    b[0] = 3*(y[1]-y[0])/(x[1]-x[0])-0.5*boundl*(x[1]-x[0]);
                }
                
                //
                // Central conditions
                //
                for(i=1; i<=n-2; i++)
                {
                    a1[i] = x[i+1]-x[i];
                    a2[i] = 2*(x[i+1]-x[i-1]);
                    a3[i] = x[i]-x[i-1];
                    b[i] = 3*(y[i]-y[i-1])/(x[i]-x[i-1])*(x[i+1]-x[i])+3*(y[i+1]-y[i])/(x[i+1]-x[i])*(x[i]-x[i-1]);
                }
                
                //
                // Right boundary conditions
                //
                if( boundrtype==0 )
                {
                    a1[n-1] = 1;
                    a2[n-1] = 1;
                    a3[n-1] = 0;
                    b[n-1] = 2*(y[n-1]-y[n-2])/(x[n-1]-x[n-2]);
                }
                if( boundrtype==1 )
                {
                    a1[n-1] = 0;
                    a2[n-1] = 1;
                    a3[n-1] = 0;
                    b[n-1] = boundr;
                }
                if( boundrtype==2 )
                {
                    a1[n-1] = 1;
                    a2[n-1] = 2;
                    a3[n-1] = 0;
                    b[n-1] = 3*(y[n-1]-y[n-2])/(x[n-1]-x[n-2])+0.5*boundr*(x[n-1]-x[n-2]);
                }
                
                //
                // Solve
                //
                solvetridiagonal(a1, a2, a3, b, n, ref d);
            }
        }


        /*************************************************************************
        Internal subroutine. Heap sort.
        *************************************************************************/
        private static void heapsortpoints(ref double[] x,
            ref double[] y,
            int n)
        {
            double[] bufx = new double[0];
            double[] bufy = new double[0];

            tsort.tagsortfastr(ref x, ref y, ref bufx, ref bufy, n);
        }


        /*************************************************************************
        Internal subroutine. Heap sort.

        Accepts:
            X, Y    -   points
            P       -   empty or preallocated array
            
        Returns:
            X, Y    -   sorted by X
            P       -   array of permutations; I-th position of output
                        arrays X/Y contains (X[P[I]],Y[P[I]])
        *************************************************************************/
        private static void heapsortppoints(ref double[] x,
            ref double[] y,
            ref int[] p,
            int n)
        {
            double[] rbuf = new double[0];
            int[] ibuf = new int[0];
            int i = 0;
            int i_ = 0;

            if( alglib.ap.len(p)<n )
            {
                p = new int[n];
            }
            rbuf = new double[n];
            for(i=0; i<=n-1; i++)
            {
                p[i] = i;
            }
            tsort.tagsortfasti(ref x, ref p, ref rbuf, ref ibuf, n);
            for(i=0; i<=n-1; i++)
            {
                rbuf[i] = y[p[i]];
            }
            for(i_=0; i_<=n-1;i_++)
            {
                y[i_] = rbuf[i_];
            }
        }


        /*************************************************************************
        Internal subroutine. Tridiagonal solver. Solves

        ( B[0] C[0]                      
        ( A[1] B[1] C[1]                 )
        (      A[2] B[2] C[2]            )
        (            ..........          ) * X = D
        (            ..........          )
        (           A[N-2] B[N-2] C[N-2] )
        (                  A[N-1] B[N-1] )

        *************************************************************************/
        private static void solvetridiagonal(double[] a,
            double[] b,
            double[] c,
            double[] d,
            int n,
            ref double[] x)
        {
            int k = 0;
            double t = 0;

            b = (double[])b.Clone();
            d = (double[])d.Clone();

            if( alglib.ap.len(x)<n )
            {
                x = new double[n];
            }
            for(k=1; k<=n-1; k++)
            {
                t = a[k]/b[k-1];
                b[k] = b[k]-t*c[k-1];
                d[k] = d[k]-t*d[k-1];
            }
            x[n-1] = d[n-1]/b[n-1];
            for(k=n-2; k>=0; k--)
            {
                x[k] = (d[k]-c[k]*x[k+1])/b[k];
            }
        }


        /*************************************************************************
        Internal subroutine. Cyclic tridiagonal solver. Solves

        ( B[0] C[0]                 A[0] )
        ( A[1] B[1] C[1]                 )
        (      A[2] B[2] C[2]            )
        (            ..........          ) * X = D
        (            ..........          )
        (           A[N-2] B[N-2] C[N-2] )
        ( C[N-1]           A[N-1] B[N-1] )
        *************************************************************************/
        private static void solvecyclictridiagonal(double[] a,
            double[] b,
            double[] c,
            double[] d,
            int n,
            ref double[] x)
        {
            int k = 0;
            double alpha = 0;
            double beta = 0;
            double gamma = 0;
            double[] y = new double[0];
            double[] z = new double[0];
            double[] u = new double[0];

            b = (double[])b.Clone();

            if( alglib.ap.len(x)<n )
            {
                x = new double[n];
            }
            beta = a[0];
            alpha = c[n-1];
            gamma = -b[0];
            b[0] = 2*b[0];
            b[n-1] = b[n-1]-alpha*beta/gamma;
            u = new double[n];
            for(k=0; k<=n-1; k++)
            {
                u[k] = 0;
            }
            u[0] = gamma;
            u[n-1] = alpha;
            solvetridiagonal(a, b, c, d, n, ref y);
            solvetridiagonal(a, b, c, u, n, ref z);
            for(k=0; k<=n-1; k++)
            {
                x[k] = y[k]-(y[0]+beta/gamma*y[n-1])/(1+z[0]+beta/gamma*z[n-1])*z[k];
            }
        }


        /*************************************************************************
        Internal subroutine. Three-point differentiation
        *************************************************************************/
        private static double diffthreepoint(double t,
            double x0,
            double f0,
            double x1,
            double f1,
            double x2,
            double f2)
        {
            double result = 0;
            double a = 0;
            double b = 0;

            t = t-x0;
            x1 = x1-x0;
            x2 = x2-x0;
            a = (f2-f0-x2/x1*(f1-f0))/(math.sqr(x2)-x1*x2);
            b = (f1-f0-a*math.sqr(x1))/x1;
            result = 2*a*t+b;
            return result;
        }


        /*************************************************************************
        Procedure for calculating value of a function is providet in the form of
        Hermite polinom  

        INPUT PARAMETERS:
            P0   -   value of a function at 0
            M0   -   value of a derivative at 0
            P1   -   value of a function at 1
            M1   -   value of a derivative at 1
            T    -   point inside [0;1]
            
        OUTPUT PARAMETERS:
            S    -   value of a function at T
            B0   -   value of a derivative function at T
            
         -- ALGLIB PROJECT --
             Copyright 26.09.2011 by Bochkanov Sergey
        *************************************************************************/
        private static void hermitecalc(double p0,
            double m0,
            double p1,
            double m1,
            double t,
            ref double s,
            ref double ds)
        {
            s = 0;
            ds = 0;

            s = p0*(1+2*t)*(1-t)*(1-t)+m0*t*(1-t)*(1-t)+p1*(3-2*t)*t*t+m1*t*t*(t-1);
            ds = -(p0*6*t*(1-t))+m0*(1-t)*(1-3*t)+p1*6*t*(1-t)+m1*t*(3*t-2);
        }


        /*************************************************************************
        Function for mapping from [A0;B0] to [A1;B1]

        INPUT PARAMETERS:
            A0   -   left border [A0;B0]
            B0   -   right border [A0;B0]
            A1   -   left border [A1;B1]
            B1   -   right border [A1;B1]
            T    -   value inside [A0;B0]  
            
        RESTRICTIONS OF PARAMETERS:

        We assume, that B0>A0 and B1>A1. But we chech, that T is inside [A0;B0], 
        and if T<A0 then T become A1, if T>B0 then T - B1. 

        INPUT PARAMETERS:
                A0   -   left border for segment [A0;B0] from 'T' is converted to [A1;B1] 
                B0   -   right border for segment [A0;B0] from 'T' is converted to [A1;B1] 
                A1   -   left border for segment [A1;B1] to 'T' is converted from [A0;B0] 
                B1   -   right border for segment [A1;B1] to 'T' is converted from [A0;B0] 
                T    -   the parameter is mapped from [A0;B0] to [A1;B1] 

        Result:
            is converted value for 'T' from [A0;B0] to [A1;B1]
                 
        REMARK:

        The function dont check value A0,B0 and A1,B1!

         -- ALGLIB PROJECT --
             Copyright 26.09.2011 by Bochkanov Sergey
        *************************************************************************/
        private static double rescaleval(double a0,
            double b0,
            double a1,
            double b1,
            double t)
        {
            double result = 0;

            
            //
            //return left border
            //
            if( (double)(t)<=(double)(a0) )
            {
                result = a1;
                return result;
            }
            
            //
            //return right border
            //
            if( (double)(t)>=(double)(b0) )
            {
                result = b1;
                return result;
            }
            
            //
            //return value between left and right borders
            //
            result = (b1-a1)*(t-a0)/(b0-a0)+a1;
            return result;
        }


    }
    public class lsfit
    {
        /*************************************************************************
        Polynomial fitting report:
            TaskRCond       reciprocal of task's condition number
            RMSError        RMS error
            AvgError        average error
            AvgRelError     average relative error (for non-zero Y[I])
            MaxError        maximum error
        *************************************************************************/
        public class polynomialfitreport
        {
            public double taskrcond;
            public double rmserror;
            public double avgerror;
            public double avgrelerror;
            public double maxerror;
        };


        /*************************************************************************
        Barycentric fitting report:
            RMSError        RMS error
            AvgError        average error
            AvgRelError     average relative error (for non-zero Y[I])
            MaxError        maximum error
            TaskRCond       reciprocal of task's condition number
        *************************************************************************/
        public class barycentricfitreport
        {
            public double taskrcond;
            public int dbest;
            public double rmserror;
            public double avgerror;
            public double avgrelerror;
            public double maxerror;
        };


        /*************************************************************************
        Spline fitting report:
            RMSError        RMS error
            AvgError        average error
            AvgRelError     average relative error (for non-zero Y[I])
            MaxError        maximum error
            
        Fields  below are  filled  by   obsolete    functions   (Spline1DFitCubic,
        Spline1DFitHermite). Modern fitting functions do NOT fill these fields:
            TaskRCond       reciprocal of task's condition number
        *************************************************************************/
        public class spline1dfitreport
        {
            public double taskrcond;
            public double rmserror;
            public double avgerror;
            public double avgrelerror;
            public double maxerror;
        };


        /*************************************************************************
        Least squares fitting report:
            TaskRCond       reciprocal of task's condition number
            IterationsCount number of internal iterations

            RMSError        RMS error
            AvgError        average error
            AvgRelError     average relative error (for non-zero Y[I])
            MaxError        maximum error

            WRMSError       weighted RMS error
        *************************************************************************/
        public class lsfitreport
        {
            public double taskrcond;
            public int iterationscount;
            public double rmserror;
            public double avgerror;
            public double avgrelerror;
            public double maxerror;
            public double wrmserror;
        };


        /*************************************************************************
        Nonlinear fitter.

        You should use ALGLIB functions to work with fitter.
        Never try to access its fields directly!
        *************************************************************************/
        public class lsfitstate
        {
            public int optalgo;
            public int m;
            public int k;
            public double epsf;
            public double epsx;
            public int maxits;
            public double stpmax;
            public bool xrep;
            public double[] s;
            public double[] bndl;
            public double[] bndu;
            public double[,] taskx;
            public double[] tasky;
            public int npoints;
            public double[] w;
            public int nweights;
            public int wkind;
            public int wits;
            public bool xupdated;
            public bool needf;
            public bool needfg;
            public bool needfgh;
            public int pointindex;
            public double[] x;
            public double[] c;
            public double f;
            public double[] g;
            public double[,] h;
            public int repiterationscount;
            public int repterminationtype;
            public double reprmserror;
            public double repavgerror;
            public double repavgrelerror;
            public double repmaxerror;
            public double repwrmserror;
            public minlm.minlmstate optstate;
            public minlm.minlmreport optrep;
            public int prevnpt;
            public int prevalgo;
            public rcommstate rstate;
            public lsfitstate()
            {
                s = new double[0];
                bndl = new double[0];
                bndu = new double[0];
                taskx = new double[0,0];
                tasky = new double[0];
                w = new double[0];
                x = new double[0];
                c = new double[0];
                g = new double[0];
                h = new double[0,0];
                optstate = new minlm.minlmstate();
                optrep = new minlm.minlmreport();
                rstate = new rcommstate();
            }
        };




        public const int rfsmax = 10;


        /*************************************************************************
        Fitting by polynomials in barycentric form. This function provides  simple
        unterface for unconstrained unweighted fitting. See  PolynomialFitWC()  if
        you need constrained fitting.

        Task is linear, so linear least squares solver is used. Complexity of this
        computational scheme is O(N*M^2), mostly dominated by least squares solver

        SEE ALSO:
            PolynomialFitWC()

        INPUT PARAMETERS:
            X   -   points, array[0..N-1].
            Y   -   function values, array[0..N-1].
            N   -   number of points, N>0
                    * if given, only leading N elements of X/Y are used
                    * if not given, automatically determined from sizes of X/Y
            M   -   number of basis functions (= polynomial_degree + 1), M>=1

        OUTPUT PARAMETERS:
            Info-   same format as in LSFitLinearW() subroutine:
                    * Info>0    task is solved
                    * Info<=0   an error occured:
                                -4 means inconvergence of internal SVD
            P   -   interpolant in barycentric form.
            Rep -   report, same format as in LSFitLinearW() subroutine.
                    Following fields are set:
                    * RMSError      rms error on the (X,Y).
                    * AvgError      average error on the (X,Y).
                    * AvgRelError   average relative error on the non-zero Y
                    * MaxError      maximum error
                                    NON-WEIGHTED ERRORS ARE CALCULATED

        NOTES:
            you can convert P from barycentric form  to  the  power  or  Chebyshev
            basis with PolynomialBar2Pow() or PolynomialBar2Cheb() functions  from
            POLINT subpackage.

          -- ALGLIB PROJECT --
             Copyright 10.12.2009 by Bochkanov Sergey
        *************************************************************************/
        public static void polynomialfit(double[] x,
            double[] y,
            int n,
            int m,
            ref int info,
            ratint.barycentricinterpolant p,
            polynomialfitreport rep)
        {
            int i = 0;
            double[] w = new double[0];
            double[] xc = new double[0];
            double[] yc = new double[0];
            int[] dc = new int[0];

            info = 0;

            alglib.ap.assert(n>0, "PolynomialFit: N<=0!");
            alglib.ap.assert(m>0, "PolynomialFit: M<=0!");
            alglib.ap.assert(alglib.ap.len(x)>=n, "PolynomialFit: Length(X)<N!");
            alglib.ap.assert(alglib.ap.len(y)>=n, "PolynomialFit: Length(Y)<N!");
            alglib.ap.assert(apserv.isfinitevector(x, n), "PolynomialFit: X contains infinite or NaN values!");
            alglib.ap.assert(apserv.isfinitevector(y, n), "PolynomialFit: Y contains infinite or NaN values!");
            w = new double[n];
            for(i=0; i<=n-1; i++)
            {
                w[i] = 1;
            }
            polynomialfitwc(x, y, w, n, xc, yc, dc, 0, m, ref info, p, rep);
        }


        /*************************************************************************
        Weighted  fitting by polynomials in barycentric form, with constraints  on
        function values or first derivatives.

        Small regularizing term is used when solving constrained tasks (to improve
        stability).

        Task is linear, so linear least squares solver is used. Complexity of this
        computational scheme is O(N*M^2), mostly dominated by least squares solver

        SEE ALSO:
            PolynomialFit()

        INPUT PARAMETERS:
            X   -   points, array[0..N-1].
            Y   -   function values, array[0..N-1].
            W   -   weights, array[0..N-1]
                    Each summand in square  sum  of  approximation deviations from
                    given  values  is  multiplied  by  the square of corresponding
                    weight. Fill it by 1's if you don't  want  to  solve  weighted
                    task.
            N   -   number of points, N>0.
                    * if given, only leading N elements of X/Y/W are used
                    * if not given, automatically determined from sizes of X/Y/W
            XC  -   points where polynomial values/derivatives are constrained,
                    array[0..K-1].
            YC  -   values of constraints, array[0..K-1]
            DC  -   array[0..K-1], types of constraints:
                    * DC[i]=0   means that P(XC[i])=YC[i]
                    * DC[i]=1   means that P'(XC[i])=YC[i]
                    SEE BELOW FOR IMPORTANT INFORMATION ON CONSTRAINTS
            K   -   number of constraints, 0<=K<M.
                    K=0 means no constraints (XC/YC/DC are not used in such cases)
            M   -   number of basis functions (= polynomial_degree + 1), M>=1

        OUTPUT PARAMETERS:
            Info-   same format as in LSFitLinearW() subroutine:
                    * Info>0    task is solved
                    * Info<=0   an error occured:
                                -4 means inconvergence of internal SVD
                                -3 means inconsistent constraints
            P   -   interpolant in barycentric form.
            Rep -   report, same format as in LSFitLinearW() subroutine.
                    Following fields are set:
                    * RMSError      rms error on the (X,Y).
                    * AvgError      average error on the (X,Y).
                    * AvgRelError   average relative error on the non-zero Y
                    * MaxError      maximum error
                                    NON-WEIGHTED ERRORS ARE CALCULATED

        IMPORTANT:
            this subroitine doesn't calculate task's condition number for K<>0.

        NOTES:
            you can convert P from barycentric form  to  the  power  or  Chebyshev
            basis with PolynomialBar2Pow() or PolynomialBar2Cheb() functions  from
            POLINT subpackage.

        SETTING CONSTRAINTS - DANGERS AND OPPORTUNITIES:

        Setting constraints can lead  to undesired  results,  like ill-conditioned
        behavior, or inconsistency being detected. From the other side,  it allows
        us to improve quality of the fit. Here we summarize  our  experience  with
        constrained regression splines:
        * even simple constraints can be inconsistent, see  Wikipedia  article  on
          this subject: http://en.wikipedia.org/wiki/Birkhoff_interpolation
        * the  greater  is  M (given  fixed  constraints),  the  more chances that
          constraints will be consistent
        * in the general case, consistency of constraints is NOT GUARANTEED.
        * in the one special cases, however, we can  guarantee  consistency.  This
          case  is:  M>1  and constraints on the function values (NOT DERIVATIVES)

        Our final recommendation is to use constraints  WHEN  AND  ONLY  when  you
        can't solve your task without them. Anything beyond  special  cases  given
        above is not guaranteed and may result in inconsistency.

          -- ALGLIB PROJECT --
             Copyright 10.12.2009 by Bochkanov Sergey
        *************************************************************************/
        public static void polynomialfitwc(double[] x,
            double[] y,
            double[] w,
            int n,
            double[] xc,
            double[] yc,
            int[] dc,
            int k,
            int m,
            ref int info,
            ratint.barycentricinterpolant p,
            polynomialfitreport rep)
        {
            double xa = 0;
            double xb = 0;
            double sa = 0;
            double sb = 0;
            double[] xoriginal = new double[0];
            double[] yoriginal = new double[0];
            double[] y2 = new double[0];
            double[] w2 = new double[0];
            double[] tmp = new double[0];
            double[] tmp2 = new double[0];
            double[] bx = new double[0];
            double[] by = new double[0];
            double[] bw = new double[0];
            int i = 0;
            int j = 0;
            double u = 0;
            double v = 0;
            double s = 0;
            int relcnt = 0;
            lsfitreport lrep = new lsfitreport();

            x = (double[])x.Clone();
            y = (double[])y.Clone();
            w = (double[])w.Clone();
            xc = (double[])xc.Clone();
            yc = (double[])yc.Clone();
            info = 0;

            alglib.ap.assert(n>0, "PolynomialFitWC: N<=0!");
            alglib.ap.assert(m>0, "PolynomialFitWC: M<=0!");
            alglib.ap.assert(k>=0, "PolynomialFitWC: K<0!");
            alglib.ap.assert(k<m, "PolynomialFitWC: K>=M!");
            alglib.ap.assert(alglib.ap.len(x)>=n, "PolynomialFitWC: Length(X)<N!");
            alglib.ap.assert(alglib.ap.len(y)>=n, "PolynomialFitWC: Length(Y)<N!");
            alglib.ap.assert(alglib.ap.len(w)>=n, "PolynomialFitWC: Length(W)<N!");
            alglib.ap.assert(alglib.ap.len(xc)>=k, "PolynomialFitWC: Length(XC)<K!");
            alglib.ap.assert(alglib.ap.len(yc)>=k, "PolynomialFitWC: Length(YC)<K!");
            alglib.ap.assert(alglib.ap.len(dc)>=k, "PolynomialFitWC: Length(DC)<K!");
            alglib.ap.assert(apserv.isfinitevector(x, n), "PolynomialFitWC: X contains infinite or NaN values!");
            alglib.ap.assert(apserv.isfinitevector(y, n), "PolynomialFitWC: Y contains infinite or NaN values!");
            alglib.ap.assert(apserv.isfinitevector(w, n), "PolynomialFitWC: X contains infinite or NaN values!");
            alglib.ap.assert(apserv.isfinitevector(xc, k), "PolynomialFitWC: XC contains infinite or NaN values!");
            alglib.ap.assert(apserv.isfinitevector(yc, k), "PolynomialFitWC: YC contains infinite or NaN values!");
            for(i=0; i<=k-1; i++)
            {
                alglib.ap.assert(dc[i]==0 || dc[i]==1, "PolynomialFitWC: one of DC[] is not 0 or 1!");
            }
            
            //
            // Scale X, Y, XC, YC.
            // Solve scaled problem using internal Chebyshev fitting function.
            //
            lsfitscalexy(ref x, ref y, ref w, n, ref xc, ref yc, dc, k, ref xa, ref xb, ref sa, ref sb, ref xoriginal, ref yoriginal);
            internalchebyshevfit(x, y, w, n, xc, yc, dc, k, m, ref info, ref tmp, lrep);
            if( info<0 )
            {
                return;
            }
            
            //
            // Generate barycentric model and scale it
            // * BX, BY store barycentric model nodes
            // * FMatrix is reused (remember - it is at least MxM, what we need)
            //
            // Model intialization is done in O(M^2). In principle, it can be
            // done in O(M*log(M)), but before it we solved task with O(N*M^2)
            // complexity, so it is only a small amount of total time spent.
            //
            bx = new double[m];
            by = new double[m];
            bw = new double[m];
            tmp2 = new double[m];
            s = 1;
            for(i=0; i<=m-1; i++)
            {
                if( m!=1 )
                {
                    u = Math.Cos(Math.PI*i/(m-1));
                }
                else
                {
                    u = 0;
                }
                v = 0;
                for(j=0; j<=m-1; j++)
                {
                    if( j==0 )
                    {
                        tmp2[j] = 1;
                    }
                    else
                    {
                        if( j==1 )
                        {
                            tmp2[j] = u;
                        }
                        else
                        {
                            tmp2[j] = 2*u*tmp2[j-1]-tmp2[j-2];
                        }
                    }
                    v = v+tmp[j]*tmp2[j];
                }
                bx[i] = u;
                by[i] = v;
                bw[i] = s;
                if( i==0 || i==m-1 )
                {
                    bw[i] = 0.5*bw[i];
                }
                s = -s;
            }
            ratint.barycentricbuildxyw(bx, by, bw, m, p);
            ratint.barycentriclintransx(p, 2/(xb-xa), -((xa+xb)/(xb-xa)));
            ratint.barycentriclintransy(p, sb-sa, sa);
            
            //
            // Scale absolute errors obtained from LSFitLinearW.
            // Relative error should be calculated separately
            // (because of shifting/scaling of the task)
            //
            rep.taskrcond = lrep.taskrcond;
            rep.rmserror = lrep.rmserror*(sb-sa);
            rep.avgerror = lrep.avgerror*(sb-sa);
            rep.maxerror = lrep.maxerror*(sb-sa);
            rep.avgrelerror = 0;
            relcnt = 0;
            for(i=0; i<=n-1; i++)
            {
                if( (double)(yoriginal[i])!=(double)(0) )
                {
                    rep.avgrelerror = rep.avgrelerror+Math.Abs(ratint.barycentriccalc(p, xoriginal[i])-yoriginal[i])/Math.Abs(yoriginal[i]);
                    relcnt = relcnt+1;
                }
            }
            if( relcnt!=0 )
            {
                rep.avgrelerror = rep.avgrelerror/relcnt;
            }
        }


        /*************************************************************************
        Weghted rational least  squares  fitting  using  Floater-Hormann  rational
        functions  with  optimal  D  chosen  from  [0,9],  with  constraints   and
        individual weights.

        Equidistant  grid  with M node on [min(x),max(x)]  is  used to build basis
        functions. Different values of D are tried, optimal D (least WEIGHTED root
        mean square error) is chosen.  Task  is  linear,  so  linear least squares
        solver  is  used.  Complexity  of  this  computational  scheme is O(N*M^2)
        (mostly dominated by the least squares solver).

        SEE ALSO
        * BarycentricFitFloaterHormann(), "lightweight" fitting without invididual
          weights and constraints.

        INPUT PARAMETERS:
            X   -   points, array[0..N-1].
            Y   -   function values, array[0..N-1].
            W   -   weights, array[0..N-1]
                    Each summand in square  sum  of  approximation deviations from
                    given  values  is  multiplied  by  the square of corresponding
                    weight. Fill it by 1's if you don't  want  to  solve  weighted
                    task.
            N   -   number of points, N>0.
            XC  -   points where function values/derivatives are constrained,
                    array[0..K-1].
            YC  -   values of constraints, array[0..K-1]
            DC  -   array[0..K-1], types of constraints:
                    * DC[i]=0   means that S(XC[i])=YC[i]
                    * DC[i]=1   means that S'(XC[i])=YC[i]
                    SEE BELOW FOR IMPORTANT INFORMATION ON CONSTRAINTS
            K   -   number of constraints, 0<=K<M.
                    K=0 means no constraints (XC/YC/DC are not used in such cases)
            M   -   number of basis functions ( = number_of_nodes), M>=2.

        OUTPUT PARAMETERS:
            Info-   same format as in LSFitLinearWC() subroutine.
                    * Info>0    task is solved
                    * Info<=0   an error occured:
                                -4 means inconvergence of internal SVD
                                -3 means inconsistent constraints
                                -1 means another errors in parameters passed
                                   (N<=0, for example)
            B   -   barycentric interpolant.
            Rep -   report, same format as in LSFitLinearWC() subroutine.
                    Following fields are set:
                    * DBest         best value of the D parameter
                    * RMSError      rms error on the (X,Y).
                    * AvgError      average error on the (X,Y).
                    * AvgRelError   average relative error on the non-zero Y
                    * MaxError      maximum error
                                    NON-WEIGHTED ERRORS ARE CALCULATED

        IMPORTANT:
            this subroutine doesn't calculate task's condition number for K<>0.

        SETTING CONSTRAINTS - DANGERS AND OPPORTUNITIES:

        Setting constraints can lead  to undesired  results,  like ill-conditioned
        behavior, or inconsistency being detected. From the other side,  it allows
        us to improve quality of the fit. Here we summarize  our  experience  with
        constrained barycentric interpolants:
        * excessive  constraints  can  be  inconsistent.   Floater-Hormann   basis
          functions aren't as flexible as splines (although they are very smooth).
        * the more evenly constraints are spread across [min(x),max(x)],  the more
          chances that they will be consistent
        * the  greater  is  M (given  fixed  constraints),  the  more chances that
          constraints will be consistent
        * in the general case, consistency of constraints IS NOT GUARANTEED.
        * in the several special cases, however, we CAN guarantee consistency.
        * one of this cases is constraints on the function  VALUES at the interval
          boundaries. Note that consustency of the  constraints  on  the  function
          DERIVATIVES is NOT guaranteed (you can use in such cases  cubic  splines
          which are more flexible).
        * another  special  case  is ONE constraint on the function value (OR, but
          not AND, derivative) anywhere in the interval

        Our final recommendation is to use constraints  WHEN  AND  ONLY  WHEN  you
        can't solve your task without them. Anything beyond  special  cases  given
        above is not guaranteed and may result in inconsistency.

          -- ALGLIB PROJECT --
             Copyright 18.08.2009 by Bochkanov Sergey
        *************************************************************************/
        public static void barycentricfitfloaterhormannwc(double[] x,
            double[] y,
            double[] w,
            int n,
            double[] xc,
            double[] yc,
            int[] dc,
            int k,
            int m,
            ref int info,
            ratint.barycentricinterpolant b,
            barycentricfitreport rep)
        {
            int d = 0;
            int i = 0;
            double wrmscur = 0;
            double wrmsbest = 0;
            ratint.barycentricinterpolant locb = new ratint.barycentricinterpolant();
            barycentricfitreport locrep = new barycentricfitreport();
            int locinfo = 0;

            info = 0;

            alglib.ap.assert(n>0, "BarycentricFitFloaterHormannWC: N<=0!");
            alglib.ap.assert(m>0, "BarycentricFitFloaterHormannWC: M<=0!");
            alglib.ap.assert(k>=0, "BarycentricFitFloaterHormannWC: K<0!");
            alglib.ap.assert(k<m, "BarycentricFitFloaterHormannWC: K>=M!");
            alglib.ap.assert(alglib.ap.len(x)>=n, "BarycentricFitFloaterHormannWC: Length(X)<N!");
            alglib.ap.assert(alglib.ap.len(y)>=n, "BarycentricFitFloaterHormannWC: Length(Y)<N!");
            alglib.ap.assert(alglib.ap.len(w)>=n, "BarycentricFitFloaterHormannWC: Length(W)<N!");
            alglib.ap.assert(alglib.ap.len(xc)>=k, "BarycentricFitFloaterHormannWC: Length(XC)<K!");
            alglib.ap.assert(alglib.ap.len(yc)>=k, "BarycentricFitFloaterHormannWC: Length(YC)<K!");
            alglib.ap.assert(alglib.ap.len(dc)>=k, "BarycentricFitFloaterHormannWC: Length(DC)<K!");
            alglib.ap.assert(apserv.isfinitevector(x, n), "BarycentricFitFloaterHormannWC: X contains infinite or NaN values!");
            alglib.ap.assert(apserv.isfinitevector(y, n), "BarycentricFitFloaterHormannWC: Y contains infinite or NaN values!");
            alglib.ap.assert(apserv.isfinitevector(w, n), "BarycentricFitFloaterHormannWC: X contains infinite or NaN values!");
            alglib.ap.assert(apserv.isfinitevector(xc, k), "BarycentricFitFloaterHormannWC: XC contains infinite or NaN values!");
            alglib.ap.assert(apserv.isfinitevector(yc, k), "BarycentricFitFloaterHormannWC: YC contains infinite or NaN values!");
            for(i=0; i<=k-1; i++)
            {
                alglib.ap.assert(dc[i]==0 || dc[i]==1, "BarycentricFitFloaterHormannWC: one of DC[] is not 0 or 1!");
            }
            
            //
            // Find optimal D
            //
            // Info is -3 by default (degenerate constraints).
            // If LocInfo will always be equal to -3, Info will remain equal to -3.
            // If at least once LocInfo will be -4, Info will be -4.
            //
            wrmsbest = math.maxrealnumber;
            rep.dbest = -1;
            info = -3;
            for(d=0; d<=Math.Min(9, n-1); d++)
            {
                barycentricfitwcfixedd(x, y, w, n, xc, yc, dc, k, m, d, ref locinfo, locb, locrep);
                alglib.ap.assert((locinfo==-4 || locinfo==-3) || locinfo>0, "BarycentricFitFloaterHormannWC: unexpected result from BarycentricFitWCFixedD!");
                if( locinfo>0 )
                {
                    
                    //
                    // Calculate weghted RMS
                    //
                    wrmscur = 0;
                    for(i=0; i<=n-1; i++)
                    {
                        wrmscur = wrmscur+math.sqr(w[i]*(y[i]-ratint.barycentriccalc(locb, x[i])));
                    }
                    wrmscur = Math.Sqrt(wrmscur/n);
                    if( (double)(wrmscur)<(double)(wrmsbest) || rep.dbest<0 )
                    {
                        ratint.barycentriccopy(locb, b);
                        rep.dbest = d;
                        info = 1;
                        rep.rmserror = locrep.rmserror;
                        rep.avgerror = locrep.avgerror;
                        rep.avgrelerror = locrep.avgrelerror;
                        rep.maxerror = locrep.maxerror;
                        rep.taskrcond = locrep.taskrcond;
                        wrmsbest = wrmscur;
                    }
                }
                else
                {
                    if( locinfo!=-3 && info<0 )
                    {
                        info = locinfo;
                    }
                }
            }
        }


        /*************************************************************************
        Rational least squares fitting using  Floater-Hormann  rational  functions
        with optimal D chosen from [0,9].

        Equidistant  grid  with M node on [min(x),max(x)]  is  used to build basis
        functions. Different values of D are tried, optimal  D  (least  root  mean
        square error) is chosen.  Task  is  linear, so linear least squares solver
        is used. Complexity  of  this  computational  scheme is  O(N*M^2)  (mostly
        dominated by the least squares solver).

        INPUT PARAMETERS:
            X   -   points, array[0..N-1].
            Y   -   function values, array[0..N-1].
            N   -   number of points, N>0.
            M   -   number of basis functions ( = number_of_nodes), M>=2.

        OUTPUT PARAMETERS:
            Info-   same format as in LSFitLinearWC() subroutine.
                    * Info>0    task is solved
                    * Info<=0   an error occured:
                                -4 means inconvergence of internal SVD
                                -3 means inconsistent constraints
            B   -   barycentric interpolant.
            Rep -   report, same format as in LSFitLinearWC() subroutine.
                    Following fields are set:
                    * DBest         best value of the D parameter
                    * RMSError      rms error on the (X,Y).
                    * AvgError      average error on the (X,Y).
                    * AvgRelError   average relative error on the non-zero Y
                    * MaxError      maximum error
                                    NON-WEIGHTED ERRORS ARE CALCULATED

          -- ALGLIB PROJECT --
             Copyright 18.08.2009 by Bochkanov Sergey
        *************************************************************************/
        public static void barycentricfitfloaterhormann(double[] x,
            double[] y,
            int n,
            int m,
            ref int info,
            ratint.barycentricinterpolant b,
            barycentricfitreport rep)
        {
            double[] w = new double[0];
            double[] xc = new double[0];
            double[] yc = new double[0];
            int[] dc = new int[0];
            int i = 0;

            info = 0;

            alglib.ap.assert(n>0, "BarycentricFitFloaterHormann: N<=0!");
            alglib.ap.assert(m>0, "BarycentricFitFloaterHormann: M<=0!");
            alglib.ap.assert(alglib.ap.len(x)>=n, "BarycentricFitFloaterHormann: Length(X)<N!");
            alglib.ap.assert(alglib.ap.len(y)>=n, "BarycentricFitFloaterHormann: Length(Y)<N!");
            alglib.ap.assert(apserv.isfinitevector(x, n), "BarycentricFitFloaterHormann: X contains infinite or NaN values!");
            alglib.ap.assert(apserv.isfinitevector(y, n), "BarycentricFitFloaterHormann: Y contains infinite or NaN values!");
            w = new double[n];
            for(i=0; i<=n-1; i++)
            {
                w[i] = 1;
            }
            barycentricfitfloaterhormannwc(x, y, w, n, xc, yc, dc, 0, m, ref info, b, rep);
        }


        /*************************************************************************
        Rational least squares fitting using  Floater-Hormann  rational  functions
        with optimal D chosen from [0,9].

        Equidistant  grid  with M node on [min(x),max(x)]  is  used to build basis
        functions. Different values of D are tried, optimal  D  (least  root  mean
        square error) is chosen.  Task  is  linear, so linear least squares solver
        is used. Complexity  of  this  computational  scheme is  O(N*M^2)  (mostly
        dominated by the least squares solver).

        INPUT PARAMETERS:
            X   -   points, array[0..N-1].
            Y   -   function values, array[0..N-1].
            N   -   number of points, N>0.
            M   -   number of basis functions ( = number_of_nodes), M>=2.

        OUTPUT PARAMETERS:
            Info-   same format as in LSFitLinearWC() subroutine.
                    * Info>0    task is solved
                    * Info<=0   an error occured:
                                -4 means inconvergence of internal SVD
                                -3 means inconsistent constraints
            B   -   barycentric interpolant.
            Rep -   report, same format as in LSFitLinearWC() subroutine.
                    Following fields are set:
                    * DBest         best value of the D parameter
                    * RMSError      rms error on the (X,Y).
                    * AvgError      average error on the (X,Y).
                    * AvgRelError   average relative error on the non-zero Y
                    * MaxError      maximum error
                                    NON-WEIGHTED ERRORS ARE CALCULATED

          -- ALGLIB PROJECT --
             Copyright 18.08.2009 by Bochkanov Sergey
        *************************************************************************/
        public static void spline1dfitpenalized(double[] x,
            double[] y,
            int n,
            int m,
            double rho,
            ref int info,
            spline1d.spline1dinterpolant s,
            spline1dfitreport rep)
        {
            double[] w = new double[0];
            int i = 0;

            x = (double[])x.Clone();
            y = (double[])y.Clone();
            info = 0;

            alglib.ap.assert(n>=1, "Spline1DFitPenalized: N<1!");
            alglib.ap.assert(m>=4, "Spline1DFitPenalized: M<4!");
            alglib.ap.assert(alglib.ap.len(x)>=n, "Spline1DFitPenalized: Length(X)<N!");
            alglib.ap.assert(alglib.ap.len(y)>=n, "Spline1DFitPenalized: Length(Y)<N!");
            alglib.ap.assert(apserv.isfinitevector(x, n), "Spline1DFitPenalized: X contains infinite or NAN values!");
            alglib.ap.assert(apserv.isfinitevector(y, n), "Spline1DFitPenalized: Y contains infinite or NAN values!");
            alglib.ap.assert(math.isfinite(rho), "Spline1DFitPenalized: Rho is infinite!");
            w = new double[n];
            for(i=0; i<=n-1; i++)
            {
                w[i] = 1;
            }
            spline1dfitpenalizedw(x, y, w, n, m, rho, ref info, s, rep);
        }


        /*************************************************************************
        Weighted fitting by penalized cubic spline.

        Equidistant grid with M nodes on [min(x,xc),max(x,xc)] is  used  to  build
        basis functions. Basis functions are cubic splines with  natural  boundary
        conditions. Problem is regularized by  adding non-linearity penalty to the
        usual least squares penalty function:

            S(x) = arg min { LS + P }, where
            LS   = SUM { w[i]^2*(y[i] - S(x[i]))^2 } - least squares penalty
            P    = C*10^rho*integral{ S''(x)^2*dx } - non-linearity penalty
            rho  - tunable constant given by user
            C    - automatically determined scale parameter,
                   makes penalty invariant with respect to scaling of X, Y, W.

        INPUT PARAMETERS:
            X   -   points, array[0..N-1].
            Y   -   function values, array[0..N-1].
            W   -   weights, array[0..N-1]
                    Each summand in square  sum  of  approximation deviations from
                    given  values  is  multiplied  by  the square of corresponding
                    weight. Fill it by 1's if you don't  want  to  solve  weighted
                    problem.
            N   -   number of points (optional):
                    * N>0
                    * if given, only first N elements of X/Y/W are processed
                    * if not given, automatically determined from X/Y/W sizes
            M   -   number of basis functions ( = number_of_nodes), M>=4.
            Rho -   regularization  constant  passed   by   user.   It   penalizes
                    nonlinearity in the regression spline. It  is  logarithmically
                    scaled,  i.e.  actual  value  of  regularization  constant  is
                    calculated as 10^Rho. It is automatically scaled so that:
                    * Rho=2.0 corresponds to moderate amount of nonlinearity
                    * generally, it should be somewhere in the [-8.0,+8.0]
                    If you do not want to penalize nonlineary,
                    pass small Rho. Values as low as -15 should work.

        OUTPUT PARAMETERS:
            Info-   same format as in LSFitLinearWC() subroutine.
                    * Info>0    task is solved
                    * Info<=0   an error occured:
                                -4 means inconvergence of internal SVD or
                                   Cholesky decomposition; problem may be
                                   too ill-conditioned (very rare)
            S   -   spline interpolant.
            Rep -   Following fields are set:
                    * RMSError      rms error on the (X,Y).
                    * AvgError      average error on the (X,Y).
                    * AvgRelError   average relative error on the non-zero Y
                    * MaxError      maximum error
                                    NON-WEIGHTED ERRORS ARE CALCULATED

        IMPORTANT:
            this subroitine doesn't calculate task's condition number for K<>0.

        NOTE 1: additional nodes are added to the spline outside  of  the  fitting
        interval to force linearity when x<min(x,xc) or x>max(x,xc).  It  is  done
        for consistency - we penalize non-linearity  at [min(x,xc),max(x,xc)],  so
        it is natural to force linearity outside of this interval.

        NOTE 2: function automatically sorts points,  so  caller may pass unsorted
        array.

          -- ALGLIB PROJECT --
             Copyright 19.10.2010 by Bochkanov Sergey
        *************************************************************************/
        public static void spline1dfitpenalizedw(double[] x,
            double[] y,
            double[] w,
            int n,
            int m,
            double rho,
            ref int info,
            spline1d.spline1dinterpolant s,
            spline1dfitreport rep)
        {
            int i = 0;
            int j = 0;
            int b = 0;
            double v = 0;
            double relcnt = 0;
            double xa = 0;
            double xb = 0;
            double sa = 0;
            double sb = 0;
            double[] xoriginal = new double[0];
            double[] yoriginal = new double[0];
            double pdecay = 0;
            double tdecay = 0;
            double[,] fmatrix = new double[0,0];
            double[] fcolumn = new double[0];
            double[] y2 = new double[0];
            double[] w2 = new double[0];
            double[] xc = new double[0];
            double[] yc = new double[0];
            int[] dc = new int[0];
            double fdmax = 0;
            double admax = 0;
            double[,] amatrix = new double[0,0];
            double[,] d2matrix = new double[0,0];
            double fa = 0;
            double ga = 0;
            double fb = 0;
            double gb = 0;
            double lambdav = 0;
            double[] bx = new double[0];
            double[] by = new double[0];
            double[] bd1 = new double[0];
            double[] bd2 = new double[0];
            double[] tx = new double[0];
            double[] ty = new double[0];
            double[] td = new double[0];
            spline1d.spline1dinterpolant bs = new spline1d.spline1dinterpolant();
            double[,] nmatrix = new double[0,0];
            double[] rightpart = new double[0];
            fbls.fblslincgstate cgstate = new fbls.fblslincgstate();
            double[] c = new double[0];
            double[] tmp0 = new double[0];
            int i_ = 0;
            int i1_ = 0;

            x = (double[])x.Clone();
            y = (double[])y.Clone();
            w = (double[])w.Clone();
            info = 0;

            alglib.ap.assert(n>=1, "Spline1DFitPenalizedW: N<1!");
            alglib.ap.assert(m>=4, "Spline1DFitPenalizedW: M<4!");
            alglib.ap.assert(alglib.ap.len(x)>=n, "Spline1DFitPenalizedW: Length(X)<N!");
            alglib.ap.assert(alglib.ap.len(y)>=n, "Spline1DFitPenalizedW: Length(Y)<N!");
            alglib.ap.assert(alglib.ap.len(w)>=n, "Spline1DFitPenalizedW: Length(W)<N!");
            alglib.ap.assert(apserv.isfinitevector(x, n), "Spline1DFitPenalizedW: X contains infinite or NAN values!");
            alglib.ap.assert(apserv.isfinitevector(y, n), "Spline1DFitPenalizedW: Y contains infinite or NAN values!");
            alglib.ap.assert(apserv.isfinitevector(w, n), "Spline1DFitPenalizedW: Y contains infinite or NAN values!");
            alglib.ap.assert(math.isfinite(rho), "Spline1DFitPenalizedW: Rho is infinite!");
            
            //
            // Prepare LambdaV
            //
            v = -(Math.Log(math.machineepsilon)/Math.Log(10));
            if( (double)(rho)<(double)(-v) )
            {
                rho = -v;
            }
            if( (double)(rho)>(double)(v) )
            {
                rho = v;
            }
            lambdav = Math.Pow(10, rho);
            
            //
            // Sort X, Y, W
            //
            spline1d.heapsortdpoints(ref x, ref y, ref w, n);
            
            //
            // Scale X, Y, XC, YC
            //
            lsfitscalexy(ref x, ref y, ref w, n, ref xc, ref yc, dc, 0, ref xa, ref xb, ref sa, ref sb, ref xoriginal, ref yoriginal);
            
            //
            // Allocate space
            //
            fmatrix = new double[n, m];
            amatrix = new double[m, m];
            d2matrix = new double[m, m];
            bx = new double[m];
            by = new double[m];
            fcolumn = new double[n];
            nmatrix = new double[m, m];
            rightpart = new double[m];
            tmp0 = new double[Math.Max(m, n)];
            c = new double[m];
            
            //
            // Fill:
            // * FMatrix by values of basis functions
            // * TmpAMatrix by second derivatives of I-th function at J-th point
            // * CMatrix by constraints
            //
            fdmax = 0;
            for(b=0; b<=m-1; b++)
            {
                
                //
                // Prepare I-th basis function
                //
                for(j=0; j<=m-1; j++)
                {
                    bx[j] = (double)(2*j)/(double)(m-1)-1;
                    by[j] = 0;
                }
                by[b] = 1;
                spline1d.spline1dgriddiff2cubic(bx, by, m, 2, 0.0, 2, 0.0, ref bd1, ref bd2);
                spline1d.spline1dbuildcubic(bx, by, m, 2, 0.0, 2, 0.0, bs);
                
                //
                // Calculate B-th column of FMatrix
                // Update FDMax (maximum column norm)
                //
                spline1d.spline1dconvcubic(bx, by, m, 2, 0.0, 2, 0.0, x, n, ref fcolumn);
                for(i_=0; i_<=n-1;i_++)
                {
                    fmatrix[i_,b] = fcolumn[i_];
                }
                v = 0;
                for(i=0; i<=n-1; i++)
                {
                    v = v+math.sqr(w[i]*fcolumn[i]);
                }
                fdmax = Math.Max(fdmax, v);
                
                //
                // Fill temporary with second derivatives of basis function
                //
                for(i_=0; i_<=m-1;i_++)
                {
                    d2matrix[b,i_] = bd2[i_];
                }
            }
            
            //
            // * calculate penalty matrix A
            // * calculate max of diagonal elements of A
            // * calculate PDecay - coefficient before penalty matrix
            //
            for(i=0; i<=m-1; i++)
            {
                for(j=i; j<=m-1; j++)
                {
                    
                    //
                    // calculate integral(B_i''*B_j'') where B_i and B_j are
                    // i-th and j-th basis splines.
                    // B_i and B_j are piecewise linear functions.
                    //
                    v = 0;
                    for(b=0; b<=m-2; b++)
                    {
                        fa = d2matrix[i,b];
                        fb = d2matrix[i,b+1];
                        ga = d2matrix[j,b];
                        gb = d2matrix[j,b+1];
                        v = v+(bx[b+1]-bx[b])*(fa*ga+(fa*(gb-ga)+ga*(fb-fa))/2+(fb-fa)*(gb-ga)/3);
                    }
                    amatrix[i,j] = v;
                    amatrix[j,i] = v;
                }
            }
            admax = 0;
            for(i=0; i<=m-1; i++)
            {
                admax = Math.Max(admax, Math.Abs(amatrix[i,i]));
            }
            pdecay = lambdav*fdmax/admax;
            
            //
            // Calculate TDecay for Tikhonov regularization
            //
            tdecay = fdmax*(1+pdecay)*10*math.machineepsilon;
            
            //
            // Prepare system
            //
            // NOTE: FMatrix is spoiled during this process
            //
            for(i=0; i<=n-1; i++)
            {
                v = w[i];
                for(i_=0; i_<=m-1;i_++)
                {
                    fmatrix[i,i_] = v*fmatrix[i,i_];
                }
            }
            ablas.rmatrixgemm(m, m, n, 1.0, fmatrix, 0, 0, 1, fmatrix, 0, 0, 0, 0.0, ref nmatrix, 0, 0);
            for(i=0; i<=m-1; i++)
            {
                for(j=0; j<=m-1; j++)
                {
                    nmatrix[i,j] = nmatrix[i,j]+pdecay*amatrix[i,j];
                }
            }
            for(i=0; i<=m-1; i++)
            {
                nmatrix[i,i] = nmatrix[i,i]+tdecay;
            }
            for(i=0; i<=m-1; i++)
            {
                rightpart[i] = 0;
            }
            for(i=0; i<=n-1; i++)
            {
                v = y[i]*w[i];
                for(i_=0; i_<=m-1;i_++)
                {
                    rightpart[i_] = rightpart[i_] + v*fmatrix[i,i_];
                }
            }
            
            //
            // Solve system
            //
            if( !trfac.spdmatrixcholesky(ref nmatrix, m, true) )
            {
                info = -4;
                return;
            }
            fbls.fblscholeskysolve(nmatrix, 1.0, m, true, ref rightpart, ref tmp0);
            for(i_=0; i_<=m-1;i_++)
            {
                c[i_] = rightpart[i_];
            }
            
            //
            // add nodes to force linearity outside of the fitting interval
            //
            spline1d.spline1dgriddiffcubic(bx, c, m, 2, 0.0, 2, 0.0, ref bd1);
            tx = new double[m+2];
            ty = new double[m+2];
            td = new double[m+2];
            i1_ = (0) - (1);
            for(i_=1; i_<=m;i_++)
            {
                tx[i_] = bx[i_+i1_];
            }
            i1_ = (0) - (1);
            for(i_=1; i_<=m;i_++)
            {
                ty[i_] = rightpart[i_+i1_];
            }
            i1_ = (0) - (1);
            for(i_=1; i_<=m;i_++)
            {
                td[i_] = bd1[i_+i1_];
            }
            tx[0] = tx[1]-(tx[2]-tx[1]);
            ty[0] = ty[1]-td[1]*(tx[2]-tx[1]);
            td[0] = td[1];
            tx[m+1] = tx[m]+(tx[m]-tx[m-1]);
            ty[m+1] = ty[m]+td[m]*(tx[m]-tx[m-1]);
            td[m+1] = td[m];
            spline1d.spline1dbuildhermite(tx, ty, td, m+2, s);
            spline1d.spline1dlintransx(s, 2/(xb-xa), -((xa+xb)/(xb-xa)));
            spline1d.spline1dlintransy(s, sb-sa, sa);
            info = 1;
            
            //
            // Fill report
            //
            rep.rmserror = 0;
            rep.avgerror = 0;
            rep.avgrelerror = 0;
            rep.maxerror = 0;
            relcnt = 0;
            spline1d.spline1dconvcubic(bx, rightpart, m, 2, 0.0, 2, 0.0, x, n, ref fcolumn);
            for(i=0; i<=n-1; i++)
            {
                v = (sb-sa)*fcolumn[i]+sa;
                rep.rmserror = rep.rmserror+math.sqr(v-yoriginal[i]);
                rep.avgerror = rep.avgerror+Math.Abs(v-yoriginal[i]);
                if( (double)(yoriginal[i])!=(double)(0) )
                {
                    rep.avgrelerror = rep.avgrelerror+Math.Abs(v-yoriginal[i])/Math.Abs(yoriginal[i]);
                    relcnt = relcnt+1;
                }
                rep.maxerror = Math.Max(rep.maxerror, Math.Abs(v-yoriginal[i]));
            }
            rep.rmserror = Math.Sqrt(rep.rmserror/n);
            rep.avgerror = rep.avgerror/n;
            if( (double)(relcnt)!=(double)(0) )
            {
                rep.avgrelerror = rep.avgrelerror/relcnt;
            }
        }


        /*************************************************************************
        Weighted fitting by cubic  spline,  with constraints on function values or
        derivatives.

        Equidistant grid with M-2 nodes on [min(x,xc),max(x,xc)] is  used to build
        basis functions. Basis functions are cubic splines with continuous  second
        derivatives  and  non-fixed first  derivatives  at  interval  ends.  Small
        regularizing term is used  when  solving  constrained  tasks  (to  improve
        stability).

        Task is linear, so linear least squares solver is used. Complexity of this
        computational scheme is O(N*M^2), mostly dominated by least squares solver

        SEE ALSO
            Spline1DFitHermiteWC()  -   fitting by Hermite splines (more flexible,
                                        less smooth)
            Spline1DFitCubic()      -   "lightweight" fitting  by  cubic  splines,
                                        without invididual weights and constraints

        INPUT PARAMETERS:
            X   -   points, array[0..N-1].
            Y   -   function values, array[0..N-1].
            W   -   weights, array[0..N-1]
                    Each summand in square  sum  of  approximation deviations from
                    given  values  is  multiplied  by  the square of corresponding
                    weight. Fill it by 1's if you don't  want  to  solve  weighted
                    task.
            N   -   number of points (optional):
                    * N>0
                    * if given, only first N elements of X/Y/W are processed
                    * if not given, automatically determined from X/Y/W sizes
            XC  -   points where spline values/derivatives are constrained,
                    array[0..K-1].
            YC  -   values of constraints, array[0..K-1]
            DC  -   array[0..K-1], types of constraints:
                    * DC[i]=0   means that S(XC[i])=YC[i]
                    * DC[i]=1   means that S'(XC[i])=YC[i]
                    SEE BELOW FOR IMPORTANT INFORMATION ON CONSTRAINTS
            K   -   number of constraints (optional):
                    * 0<=K<M.
                    * K=0 means no constraints (XC/YC/DC are not used)
                    * if given, only first K elements of XC/YC/DC are used
                    * if not given, automatically determined from XC/YC/DC
            M   -   number of basis functions ( = number_of_nodes+2), M>=4.

        OUTPUT PARAMETERS:
            Info-   same format as in LSFitLinearWC() subroutine.
                    * Info>0    task is solved
                    * Info<=0   an error occured:
                                -4 means inconvergence of internal SVD
                                -3 means inconsistent constraints
            S   -   spline interpolant.
            Rep -   report, same format as in LSFitLinearWC() subroutine.
                    Following fields are set:
                    * RMSError      rms error on the (X,Y).
                    * AvgError      average error on the (X,Y).
                    * AvgRelError   average relative error on the non-zero Y
                    * MaxError      maximum error
                                    NON-WEIGHTED ERRORS ARE CALCULATED

        IMPORTANT:
            this subroitine doesn't calculate task's condition number for K<>0.


        ORDER OF POINTS

        Subroutine automatically sorts points, so caller may pass unsorted array.

        SETTING CONSTRAINTS - DANGERS AND OPPORTUNITIES:

        Setting constraints can lead  to undesired  results,  like ill-conditioned
        behavior, or inconsistency being detected. From the other side,  it allows
        us to improve quality of the fit. Here we summarize  our  experience  with
        constrained regression splines:
        * excessive constraints can be inconsistent. Splines are  piecewise  cubic
          functions, and it is easy to create an example, where  large  number  of
          constraints  concentrated  in  small  area will result in inconsistency.
          Just because spline is not flexible enough to satisfy all of  them.  And
          same constraints spread across the  [min(x),max(x)]  will  be  perfectly
          consistent.
        * the more evenly constraints are spread across [min(x),max(x)],  the more
          chances that they will be consistent
        * the  greater  is  M (given  fixed  constraints),  the  more chances that
          constraints will be consistent
        * in the general case, consistency of constraints IS NOT GUARANTEED.
        * in the several special cases, however, we CAN guarantee consistency.
        * one of this cases is constraints  on  the  function  values  AND/OR  its
          derivatives at the interval boundaries.
        * another  special  case  is ONE constraint on the function value (OR, but
          not AND, derivative) anywhere in the interval

        Our final recommendation is to use constraints  WHEN  AND  ONLY  WHEN  you
        can't solve your task without them. Anything beyond  special  cases  given
        above is not guaranteed and may result in inconsistency.


          -- ALGLIB PROJECT --
             Copyright 18.08.2009 by Bochkanov Sergey
        *************************************************************************/
        public static void spline1dfitcubicwc(double[] x,
            double[] y,
            double[] w,
            int n,
            double[] xc,
            double[] yc,
            int[] dc,
            int k,
            int m,
            ref int info,
            spline1d.spline1dinterpolant s,
            spline1dfitreport rep)
        {
            int i = 0;

            info = 0;

            alglib.ap.assert(n>=1, "Spline1DFitCubicWC: N<1!");
            alglib.ap.assert(m>=4, "Spline1DFitCubicWC: M<4!");
            alglib.ap.assert(k>=0, "Spline1DFitCubicWC: K<0!");
            alglib.ap.assert(k<m, "Spline1DFitCubicWC: K>=M!");
            alglib.ap.assert(alglib.ap.len(x)>=n, "Spline1DFitCubicWC: Length(X)<N!");
            alglib.ap.assert(alglib.ap.len(y)>=n, "Spline1DFitCubicWC: Length(Y)<N!");
            alglib.ap.assert(alglib.ap.len(w)>=n, "Spline1DFitCubicWC: Length(W)<N!");
            alglib.ap.assert(alglib.ap.len(xc)>=k, "Spline1DFitCubicWC: Length(XC)<K!");
            alglib.ap.assert(alglib.ap.len(yc)>=k, "Spline1DFitCubicWC: Length(YC)<K!");
            alglib.ap.assert(alglib.ap.len(dc)>=k, "Spline1DFitCubicWC: Length(DC)<K!");
            alglib.ap.assert(apserv.isfinitevector(x, n), "Spline1DFitCubicWC: X contains infinite or NAN values!");
            alglib.ap.assert(apserv.isfinitevector(y, n), "Spline1DFitCubicWC: Y contains infinite or NAN values!");
            alglib.ap.assert(apserv.isfinitevector(w, n), "Spline1DFitCubicWC: Y contains infinite or NAN values!");
            alglib.ap.assert(apserv.isfinitevector(xc, k), "Spline1DFitCubicWC: X contains infinite or NAN values!");
            alglib.ap.assert(apserv.isfinitevector(yc, k), "Spline1DFitCubicWC: Y contains infinite or NAN values!");
            for(i=0; i<=k-1; i++)
            {
                alglib.ap.assert(dc[i]==0 || dc[i]==1, "Spline1DFitCubicWC: DC[i] is neither 0 or 1!");
            }
            spline1dfitinternal(0, x, y, w, n, xc, yc, dc, k, m, ref info, s, rep);
        }


        /*************************************************************************
        Weighted  fitting  by Hermite spline,  with constraints on function values
        or first derivatives.

        Equidistant grid with M nodes on [min(x,xc),max(x,xc)] is  used  to  build
        basis functions. Basis functions are Hermite splines.  Small  regularizing
        term is used when solving constrained tasks (to improve stability).

        Task is linear, so linear least squares solver is used. Complexity of this
        computational scheme is O(N*M^2), mostly dominated by least squares solver

        SEE ALSO
            Spline1DFitCubicWC()    -   fitting by Cubic splines (less flexible,
                                        more smooth)
            Spline1DFitHermite()    -   "lightweight" Hermite fitting, without
                                        invididual weights and constraints

        INPUT PARAMETERS:
            X   -   points, array[0..N-1].
            Y   -   function values, array[0..N-1].
            W   -   weights, array[0..N-1]
                    Each summand in square  sum  of  approximation deviations from
                    given  values  is  multiplied  by  the square of corresponding
                    weight. Fill it by 1's if you don't  want  to  solve  weighted
                    task.
            N   -   number of points (optional):
                    * N>0
                    * if given, only first N elements of X/Y/W are processed
                    * if not given, automatically determined from X/Y/W sizes
            XC  -   points where spline values/derivatives are constrained,
                    array[0..K-1].
            YC  -   values of constraints, array[0..K-1]
            DC  -   array[0..K-1], types of constraints:
                    * DC[i]=0   means that S(XC[i])=YC[i]
                    * DC[i]=1   means that S'(XC[i])=YC[i]
                    SEE BELOW FOR IMPORTANT INFORMATION ON CONSTRAINTS
            K   -   number of constraints (optional):
                    * 0<=K<M.
                    * K=0 means no constraints (XC/YC/DC are not used)
                    * if given, only first K elements of XC/YC/DC are used
                    * if not given, automatically determined from XC/YC/DC
            M   -   number of basis functions (= 2 * number of nodes),
                    M>=4,
                    M IS EVEN!

        OUTPUT PARAMETERS:
            Info-   same format as in LSFitLinearW() subroutine:
                    * Info>0    task is solved
                    * Info<=0   an error occured:
                                -4 means inconvergence of internal SVD
                                -3 means inconsistent constraints
                                -2 means odd M was passed (which is not supported)
                                -1 means another errors in parameters passed
                                   (N<=0, for example)
            S   -   spline interpolant.
            Rep -   report, same format as in LSFitLinearW() subroutine.
                    Following fields are set:
                    * RMSError      rms error on the (X,Y).
                    * AvgError      average error on the (X,Y).
                    * AvgRelError   average relative error on the non-zero Y
                    * MaxError      maximum error
                                    NON-WEIGHTED ERRORS ARE CALCULATED

        IMPORTANT:
            this subroitine doesn't calculate task's condition number for K<>0.

        IMPORTANT:
            this subroitine supports only even M's


        ORDER OF POINTS

        Subroutine automatically sorts points, so caller may pass unsorted array.

        SETTING CONSTRAINTS - DANGERS AND OPPORTUNITIES:

        Setting constraints can lead  to undesired  results,  like ill-conditioned
        behavior, or inconsistency being detected. From the other side,  it allows
        us to improve quality of the fit. Here we summarize  our  experience  with
        constrained regression splines:
        * excessive constraints can be inconsistent. Splines are  piecewise  cubic
          functions, and it is easy to create an example, where  large  number  of
          constraints  concentrated  in  small  area will result in inconsistency.
          Just because spline is not flexible enough to satisfy all of  them.  And
          same constraints spread across the  [min(x),max(x)]  will  be  perfectly
          consistent.
        * the more evenly constraints are spread across [min(x),max(x)],  the more
          chances that they will be consistent
        * the  greater  is  M (given  fixed  constraints),  the  more chances that
          constraints will be consistent
        * in the general case, consistency of constraints is NOT GUARANTEED.
        * in the several special cases, however, we can guarantee consistency.
        * one of this cases is  M>=4  and   constraints  on   the  function  value
          (AND/OR its derivative) at the interval boundaries.
        * another special case is M>=4  and  ONE  constraint on the function value
          (OR, BUT NOT AND, derivative) anywhere in [min(x),max(x)]

        Our final recommendation is to use constraints  WHEN  AND  ONLY  when  you
        can't solve your task without them. Anything beyond  special  cases  given
        above is not guaranteed and may result in inconsistency.

          -- ALGLIB PROJECT --
             Copyright 18.08.2009 by Bochkanov Sergey
        *************************************************************************/
        public static void spline1dfithermitewc(double[] x,
            double[] y,
            double[] w,
            int n,
            double[] xc,
            double[] yc,
            int[] dc,
            int k,
            int m,
            ref int info,
            spline1d.spline1dinterpolant s,
            spline1dfitreport rep)
        {
            int i = 0;

            info = 0;

            alglib.ap.assert(n>=1, "Spline1DFitHermiteWC: N<1!");
            alglib.ap.assert(m>=4, "Spline1DFitHermiteWC: M<4!");
            alglib.ap.assert(m%2==0, "Spline1DFitHermiteWC: M is odd!");
            alglib.ap.assert(k>=0, "Spline1DFitHermiteWC: K<0!");
            alglib.ap.assert(k<m, "Spline1DFitHermiteWC: K>=M!");
            alglib.ap.assert(alglib.ap.len(x)>=n, "Spline1DFitHermiteWC: Length(X)<N!");
            alglib.ap.assert(alglib.ap.len(y)>=n, "Spline1DFitHermiteWC: Length(Y)<N!");
            alglib.ap.assert(alglib.ap.len(w)>=n, "Spline1DFitHermiteWC: Length(W)<N!");
            alglib.ap.assert(alglib.ap.len(xc)>=k, "Spline1DFitHermiteWC: Length(XC)<K!");
            alglib.ap.assert(alglib.ap.len(yc)>=k, "Spline1DFitHermiteWC: Length(YC)<K!");
            alglib.ap.assert(alglib.ap.len(dc)>=k, "Spline1DFitHermiteWC: Length(DC)<K!");
            alglib.ap.assert(apserv.isfinitevector(x, n), "Spline1DFitHermiteWC: X contains infinite or NAN values!");
            alglib.ap.assert(apserv.isfinitevector(y, n), "Spline1DFitHermiteWC: Y contains infinite or NAN values!");
            alglib.ap.assert(apserv.isfinitevector(w, n), "Spline1DFitHermiteWC: Y contains infinite or NAN values!");
            alglib.ap.assert(apserv.isfinitevector(xc, k), "Spline1DFitHermiteWC: X contains infinite or NAN values!");
            alglib.ap.assert(apserv.isfinitevector(yc, k), "Spline1DFitHermiteWC: Y contains infinite or NAN values!");
            for(i=0; i<=k-1; i++)
            {
                alglib.ap.assert(dc[i]==0 || dc[i]==1, "Spline1DFitHermiteWC: DC[i] is neither 0 or 1!");
            }
            spline1dfitinternal(1, x, y, w, n, xc, yc, dc, k, m, ref info, s, rep);
        }


        /*************************************************************************
        Least squares fitting by cubic spline.

        This subroutine is "lightweight" alternative for more complex and feature-
        rich Spline1DFitCubicWC().  See  Spline1DFitCubicWC() for more information
        about subroutine parameters (we don't duplicate it here because of length)

          -- ALGLIB PROJECT --
             Copyright 18.08.2009 by Bochkanov Sergey
        *************************************************************************/
        public static void spline1dfitcubic(double[] x,
            double[] y,
            int n,
            int m,
            ref int info,
            spline1d.spline1dinterpolant s,
            spline1dfitreport rep)
        {
            int i = 0;
            double[] w = new double[0];
            double[] xc = new double[0];
            double[] yc = new double[0];
            int[] dc = new int[0];

            info = 0;

            alglib.ap.assert(n>=1, "Spline1DFitCubic: N<1!");
            alglib.ap.assert(m>=4, "Spline1DFitCubic: M<4!");
            alglib.ap.assert(alglib.ap.len(x)>=n, "Spline1DFitCubic: Length(X)<N!");
            alglib.ap.assert(alglib.ap.len(y)>=n, "Spline1DFitCubic: Length(Y)<N!");
            alglib.ap.assert(apserv.isfinitevector(x, n), "Spline1DFitCubic: X contains infinite or NAN values!");
            alglib.ap.assert(apserv.isfinitevector(y, n), "Spline1DFitCubic: Y contains infinite or NAN values!");
            w = new double[n];
            for(i=0; i<=n-1; i++)
            {
                w[i] = 1;
            }
            spline1dfitcubicwc(x, y, w, n, xc, yc, dc, 0, m, ref info, s, rep);
        }


        /*************************************************************************
        Least squares fitting by Hermite spline.

        This subroutine is "lightweight" alternative for more complex and feature-
        rich Spline1DFitHermiteWC().  See Spline1DFitHermiteWC()  description  for
        more information about subroutine parameters (we don't duplicate  it  here
        because of length).

          -- ALGLIB PROJECT --
             Copyright 18.08.2009 by Bochkanov Sergey
        *************************************************************************/
        public static void spline1dfithermite(double[] x,
            double[] y,
            int n,
            int m,
            ref int info,
            spline1d.spline1dinterpolant s,
            spline1dfitreport rep)
        {
            int i = 0;
            double[] w = new double[0];
            double[] xc = new double[0];
            double[] yc = new double[0];
            int[] dc = new int[0];

            info = 0;

            alglib.ap.assert(n>=1, "Spline1DFitHermite: N<1!");
            alglib.ap.assert(m>=4, "Spline1DFitHermite: M<4!");
            alglib.ap.assert(m%2==0, "Spline1DFitHermite: M is odd!");
            alglib.ap.assert(alglib.ap.len(x)>=n, "Spline1DFitHermite: Length(X)<N!");
            alglib.ap.assert(alglib.ap.len(y)>=n, "Spline1DFitHermite: Length(Y)<N!");
            alglib.ap.assert(apserv.isfinitevector(x, n), "Spline1DFitHermite: X contains infinite or NAN values!");
            alglib.ap.assert(apserv.isfinitevector(y, n), "Spline1DFitHermite: Y contains infinite or NAN values!");
            w = new double[n];
            for(i=0; i<=n-1; i++)
            {
                w[i] = 1;
            }
            spline1dfithermitewc(x, y, w, n, xc, yc, dc, 0, m, ref info, s, rep);
        }


        /*************************************************************************
        Weighted linear least squares fitting.

        QR decomposition is used to reduce task to MxM, then triangular solver  or
        SVD-based solver is used depending on condition number of the  system.  It
        allows to maximize speed and retain decent accuracy.

        INPUT PARAMETERS:
            Y       -   array[0..N-1] Function values in  N  points.
            W       -   array[0..N-1]  Weights  corresponding to function  values.
                        Each summand in square  sum  of  approximation  deviations
                        from  given  values  is  multiplied  by  the   square   of
                        corresponding weight.
            FMatrix -   a table of basis functions values, array[0..N-1, 0..M-1].
                        FMatrix[I, J] - value of J-th basis function in I-th point.
            N       -   number of points used. N>=1.
            M       -   number of basis functions, M>=1.

        OUTPUT PARAMETERS:
            Info    -   error code:
                        * -4    internal SVD decomposition subroutine failed (very
                                rare and for degenerate systems only)
                        * -1    incorrect N/M were specified
                        *  1    task is solved
            C       -   decomposition coefficients, array[0..M-1]
            Rep     -   fitting report. Following fields are set:
                        * Rep.TaskRCond     reciprocal of condition number
                        * RMSError          rms error on the (X,Y).
                        * AvgError          average error on the (X,Y).
                        * AvgRelError       average relative error on the non-zero Y
                        * MaxError          maximum error
                                            NON-WEIGHTED ERRORS ARE CALCULATED

          -- ALGLIB --
             Copyright 17.08.2009 by Bochkanov Sergey
        *************************************************************************/
        public static void lsfitlinearw(double[] y,
            double[] w,
            double[,] fmatrix,
            int n,
            int m,
            ref int info,
            ref double[] c,
            lsfitreport rep)
        {
            info = 0;
            c = new double[0];

            alglib.ap.assert(n>=1, "LSFitLinearW: N<1!");
            alglib.ap.assert(m>=1, "LSFitLinearW: M<1!");
            alglib.ap.assert(alglib.ap.len(y)>=n, "LSFitLinearW: length(Y)<N!");
            alglib.ap.assert(apserv.isfinitevector(y, n), "LSFitLinearW: Y contains infinite or NaN values!");
            alglib.ap.assert(alglib.ap.len(w)>=n, "LSFitLinearW: length(W)<N!");
            alglib.ap.assert(apserv.isfinitevector(w, n), "LSFitLinearW: W contains infinite or NaN values!");
            alglib.ap.assert(alglib.ap.rows(fmatrix)>=n, "LSFitLinearW: rows(FMatrix)<N!");
            alglib.ap.assert(alglib.ap.cols(fmatrix)>=m, "LSFitLinearW: cols(FMatrix)<M!");
            alglib.ap.assert(apserv.apservisfinitematrix(fmatrix, n, m), "LSFitLinearW: FMatrix contains infinite or NaN values!");
            lsfitlinearinternal(y, w, fmatrix, n, m, ref info, ref c, rep);
        }


        /*************************************************************************
        Weighted constained linear least squares fitting.

        This  is  variation  of LSFitLinearW(), which searchs for min|A*x=b| given
        that  K  additional  constaints  C*x=bc are satisfied. It reduces original
        task to modified one: min|B*y-d| WITHOUT constraints,  then LSFitLinearW()
        is called.

        INPUT PARAMETERS:
            Y       -   array[0..N-1] Function values in  N  points.
            W       -   array[0..N-1]  Weights  corresponding to function  values.
                        Each summand in square  sum  of  approximation  deviations
                        from  given  values  is  multiplied  by  the   square   of
                        corresponding weight.
            FMatrix -   a table of basis functions values, array[0..N-1, 0..M-1].
                        FMatrix[I,J] - value of J-th basis function in I-th point.
            CMatrix -   a table of constaints, array[0..K-1,0..M].
                        I-th row of CMatrix corresponds to I-th linear constraint:
                        CMatrix[I,0]*C[0] + ... + CMatrix[I,M-1]*C[M-1] = CMatrix[I,M]
            N       -   number of points used. N>=1.
            M       -   number of basis functions, M>=1.
            K       -   number of constraints, 0 <= K < M
                        K=0 corresponds to absence of constraints.

        OUTPUT PARAMETERS:
            Info    -   error code:
                        * -4    internal SVD decomposition subroutine failed (very
                                rare and for degenerate systems only)
                        * -3    either   too   many  constraints  (M   or   more),
                                degenerate  constraints   (some   constraints  are
                                repetead twice) or inconsistent  constraints  were
                                specified.
                        *  1    task is solved
            C       -   decomposition coefficients, array[0..M-1]
            Rep     -   fitting report. Following fields are set:
                        * RMSError          rms error on the (X,Y).
                        * AvgError          average error on the (X,Y).
                        * AvgRelError       average relative error on the non-zero Y
                        * MaxError          maximum error
                                            NON-WEIGHTED ERRORS ARE CALCULATED

        IMPORTANT:
            this subroitine doesn't calculate task's condition number for K<>0.

          -- ALGLIB --
             Copyright 07.09.2009 by Bochkanov Sergey
        *************************************************************************/
        public static void lsfitlinearwc(double[] y,
            double[] w,
            double[,] fmatrix,
            double[,] cmatrix,
            int n,
            int m,
            int k,
            ref int info,
            ref double[] c,
            lsfitreport rep)
        {
            int i = 0;
            int j = 0;
            double[] tau = new double[0];
            double[,] q = new double[0,0];
            double[,] f2 = new double[0,0];
            double[] tmp = new double[0];
            double[] c0 = new double[0];
            double v = 0;
            int i_ = 0;

            y = (double[])y.Clone();
            cmatrix = (double[,])cmatrix.Clone();
            info = 0;
            c = new double[0];

            alglib.ap.assert(n>=1, "LSFitLinearWC: N<1!");
            alglib.ap.assert(m>=1, "LSFitLinearWC: M<1!");
            alglib.ap.assert(k>=0, "LSFitLinearWC: K<0!");
            alglib.ap.assert(alglib.ap.len(y)>=n, "LSFitLinearWC: length(Y)<N!");
            alglib.ap.assert(apserv.isfinitevector(y, n), "LSFitLinearWC: Y contains infinite or NaN values!");
            alglib.ap.assert(alglib.ap.len(w)>=n, "LSFitLinearWC: length(W)<N!");
            alglib.ap.assert(apserv.isfinitevector(w, n), "LSFitLinearWC: W contains infinite or NaN values!");
            alglib.ap.assert(alglib.ap.rows(fmatrix)>=n, "LSFitLinearWC: rows(FMatrix)<N!");
            alglib.ap.assert(alglib.ap.cols(fmatrix)>=m, "LSFitLinearWC: cols(FMatrix)<M!");
            alglib.ap.assert(apserv.apservisfinitematrix(fmatrix, n, m), "LSFitLinearWC: FMatrix contains infinite or NaN values!");
            alglib.ap.assert(alglib.ap.rows(cmatrix)>=k, "LSFitLinearWC: rows(CMatrix)<K!");
            alglib.ap.assert(alglib.ap.cols(cmatrix)>=m+1 || k==0, "LSFitLinearWC: cols(CMatrix)<M+1!");
            alglib.ap.assert(apserv.apservisfinitematrix(cmatrix, k, m+1), "LSFitLinearWC: CMatrix contains infinite or NaN values!");
            if( k>=m )
            {
                info = -3;
                return;
            }
            
            //
            // Solve
            //
            if( k==0 )
            {
                
                //
                // no constraints
                //
                lsfitlinearinternal(y, w, fmatrix, n, m, ref info, ref c, rep);
            }
            else
            {
                
                //
                // First, find general form solution of constraints system:
                // * factorize C = L*Q
                // * unpack Q
                // * fill upper part of C with zeros (for RCond)
                //
                // We got C=C0+Q2'*y where Q2 is lower M-K rows of Q.
                //
                ortfac.rmatrixlq(ref cmatrix, k, m, ref tau);
                ortfac.rmatrixlqunpackq(cmatrix, k, m, tau, m, ref q);
                for(i=0; i<=k-1; i++)
                {
                    for(j=i+1; j<=m-1; j++)
                    {
                        cmatrix[i,j] = 0.0;
                    }
                }
                if( (double)(rcond.rmatrixlurcondinf(cmatrix, k))<(double)(1000*math.machineepsilon) )
                {
                    info = -3;
                    return;
                }
                tmp = new double[k];
                for(i=0; i<=k-1; i++)
                {
                    if( i>0 )
                    {
                        v = 0.0;
                        for(i_=0; i_<=i-1;i_++)
                        {
                            v += cmatrix[i,i_]*tmp[i_];
                        }
                    }
                    else
                    {
                        v = 0;
                    }
                    tmp[i] = (cmatrix[i,m]-v)/cmatrix[i,i];
                }
                c0 = new double[m];
                for(i=0; i<=m-1; i++)
                {
                    c0[i] = 0;
                }
                for(i=0; i<=k-1; i++)
                {
                    v = tmp[i];
                    for(i_=0; i_<=m-1;i_++)
                    {
                        c0[i_] = c0[i_] + v*q[i,i_];
                    }
                }
                
                //
                // Second, prepare modified matrix F2 = F*Q2' and solve modified task
                //
                tmp = new double[Math.Max(n, m)+1];
                f2 = new double[n, m-k];
                blas.matrixvectormultiply(fmatrix, 0, n-1, 0, m-1, false, c0, 0, m-1, -1.0, ref y, 0, n-1, 1.0);
                blas.matrixmatrixmultiply(fmatrix, 0, n-1, 0, m-1, false, q, k, m-1, 0, m-1, true, 1.0, ref f2, 0, n-1, 0, m-k-1, 0.0, ref tmp);
                lsfitlinearinternal(y, w, f2, n, m-k, ref info, ref tmp, rep);
                rep.taskrcond = -1;
                if( info<=0 )
                {
                    return;
                }
                
                //
                // then, convert back to original answer: C = C0 + Q2'*Y0
                //
                c = new double[m];
                for(i_=0; i_<=m-1;i_++)
                {
                    c[i_] = c0[i_];
                }
                blas.matrixvectormultiply(q, k, m-1, 0, m-1, true, tmp, 0, m-k-1, 1.0, ref c, 0, m-1, 1.0);
            }
        }


        /*************************************************************************
        Linear least squares fitting.

        QR decomposition is used to reduce task to MxM, then triangular solver  or
        SVD-based solver is used depending on condition number of the  system.  It
        allows to maximize speed and retain decent accuracy.

        INPUT PARAMETERS:
            Y       -   array[0..N-1] Function values in  N  points.
            FMatrix -   a table of basis functions values, array[0..N-1, 0..M-1].
                        FMatrix[I, J] - value of J-th basis function in I-th point.
            N       -   number of points used. N>=1.
            M       -   number of basis functions, M>=1.

        OUTPUT PARAMETERS:
            Info    -   error code:
                        * -4    internal SVD decomposition subroutine failed (very
                                rare and for degenerate systems only)
                        *  1    task is solved
            C       -   decomposition coefficients, array[0..M-1]
            Rep     -   fitting report. Following fields are set:
                        * Rep.TaskRCond     reciprocal of condition number
                        * RMSError          rms error on the (X,Y).
                        * AvgError          average error on the (X,Y).
                        * AvgRelError       average relative error on the non-zero Y
                        * MaxError          maximum error
                                            NON-WEIGHTED ERRORS ARE CALCULATED

          -- ALGLIB --
             Copyright 17.08.2009 by Bochkanov Sergey
        *************************************************************************/
        public static void lsfitlinear(double[] y,
            double[,] fmatrix,
            int n,
            int m,
            ref int info,
            ref double[] c,
            lsfitreport rep)
        {
            double[] w = new double[0];
            int i = 0;

            info = 0;
            c = new double[0];

            alglib.ap.assert(n>=1, "LSFitLinear: N<1!");
            alglib.ap.assert(m>=1, "LSFitLinear: M<1!");
            alglib.ap.assert(alglib.ap.len(y)>=n, "LSFitLinear: length(Y)<N!");
            alglib.ap.assert(apserv.isfinitevector(y, n), "LSFitLinear: Y contains infinite or NaN values!");
            alglib.ap.assert(alglib.ap.rows(fmatrix)>=n, "LSFitLinear: rows(FMatrix)<N!");
            alglib.ap.assert(alglib.ap.cols(fmatrix)>=m, "LSFitLinear: cols(FMatrix)<M!");
            alglib.ap.assert(apserv.apservisfinitematrix(fmatrix, n, m), "LSFitLinear: FMatrix contains infinite or NaN values!");
            w = new double[n];
            for(i=0; i<=n-1; i++)
            {
                w[i] = 1;
            }
            lsfitlinearinternal(y, w, fmatrix, n, m, ref info, ref c, rep);
        }


        /*************************************************************************
        Constained linear least squares fitting.

        This  is  variation  of LSFitLinear(),  which searchs for min|A*x=b| given
        that  K  additional  constaints  C*x=bc are satisfied. It reduces original
        task to modified one: min|B*y-d| WITHOUT constraints,  then  LSFitLinear()
        is called.

        INPUT PARAMETERS:
            Y       -   array[0..N-1] Function values in  N  points.
            FMatrix -   a table of basis functions values, array[0..N-1, 0..M-1].
                        FMatrix[I,J] - value of J-th basis function in I-th point.
            CMatrix -   a table of constaints, array[0..K-1,0..M].
                        I-th row of CMatrix corresponds to I-th linear constraint:
                        CMatrix[I,0]*C[0] + ... + CMatrix[I,M-1]*C[M-1] = CMatrix[I,M]
            N       -   number of points used. N>=1.
            M       -   number of basis functions, M>=1.
            K       -   number of constraints, 0 <= K < M
                        K=0 corresponds to absence of constraints.

        OUTPUT PARAMETERS:
            Info    -   error code:
                        * -4    internal SVD decomposition subroutine failed (very
                                rare and for degenerate systems only)
                        * -3    either   too   many  constraints  (M   or   more),
                                degenerate  constraints   (some   constraints  are
                                repetead twice) or inconsistent  constraints  were
                                specified.
                        *  1    task is solved
            C       -   decomposition coefficients, array[0..M-1]
            Rep     -   fitting report. Following fields are set:
                        * RMSError          rms error on the (X,Y).
                        * AvgError          average error on the (X,Y).
                        * AvgRelError       average relative error on the non-zero Y
                        * MaxError          maximum error
                                            NON-WEIGHTED ERRORS ARE CALCULATED

        IMPORTANT:
            this subroitine doesn't calculate task's condition number for K<>0.

          -- ALGLIB --
             Copyright 07.09.2009 by Bochkanov Sergey
        *************************************************************************/
        public static void lsfitlinearc(double[] y,
            double[,] fmatrix,
            double[,] cmatrix,
            int n,
            int m,
            int k,
            ref int info,
            ref double[] c,
            lsfitreport rep)
        {
            double[] w = new double[0];
            int i = 0;

            y = (double[])y.Clone();
            info = 0;
            c = new double[0];

            alglib.ap.assert(n>=1, "LSFitLinearC: N<1!");
            alglib.ap.assert(m>=1, "LSFitLinearC: M<1!");
            alglib.ap.assert(k>=0, "LSFitLinearC: K<0!");
            alglib.ap.assert(alglib.ap.len(y)>=n, "LSFitLinearC: length(Y)<N!");
            alglib.ap.assert(apserv.isfinitevector(y, n), "LSFitLinearC: Y contains infinite or NaN values!");
            alglib.ap.assert(alglib.ap.rows(fmatrix)>=n, "LSFitLinearC: rows(FMatrix)<N!");
            alglib.ap.assert(alglib.ap.cols(fmatrix)>=m, "LSFitLinearC: cols(FMatrix)<M!");
            alglib.ap.assert(apserv.apservisfinitematrix(fmatrix, n, m), "LSFitLinearC: FMatrix contains infinite or NaN values!");
            alglib.ap.assert(alglib.ap.rows(cmatrix)>=k, "LSFitLinearC: rows(CMatrix)<K!");
            alglib.ap.assert(alglib.ap.cols(cmatrix)>=m+1 || k==0, "LSFitLinearC: cols(CMatrix)<M+1!");
            alglib.ap.assert(apserv.apservisfinitematrix(cmatrix, k, m+1), "LSFitLinearC: CMatrix contains infinite or NaN values!");
            w = new double[n];
            for(i=0; i<=n-1; i++)
            {
                w[i] = 1;
            }
            lsfitlinearwc(y, w, fmatrix, cmatrix, n, m, k, ref info, ref c, rep);
        }


        /*************************************************************************
        Weighted nonlinear least squares fitting using function values only.

        Combination of numerical differentiation and secant updates is used to
        obtain function Jacobian.

        Nonlinear task min(F(c)) is solved, where

            F(c) = (w[0]*(f(c,x[0])-y[0]))^2 + ... + (w[n-1]*(f(c,x[n-1])-y[n-1]))^2,

            * N is a number of points,
            * M is a dimension of a space points belong to,
            * K is a dimension of a space of parameters being fitted,
            * w is an N-dimensional vector of weight coefficients,
            * x is a set of N points, each of them is an M-dimensional vector,
            * c is a K-dimensional vector of parameters being fitted

        This subroutine uses only f(c,x[i]).

        INPUT PARAMETERS:
            X       -   array[0..N-1,0..M-1], points (one row = one point)
            Y       -   array[0..N-1], function values.
            W       -   weights, array[0..N-1]
            C       -   array[0..K-1], initial approximation to the solution,
            N       -   number of points, N>1
            M       -   dimension of space
            K       -   number of parameters being fitted
            DiffStep-   numerical differentiation step;
                        should not be very small or large;
                        large = loss of accuracy
                        small = growth of round-off errors

        OUTPUT PARAMETERS:
            State   -   structure which stores algorithm state

          -- ALGLIB --
             Copyright 18.10.2008 by Bochkanov Sergey
        *************************************************************************/
        public static void lsfitcreatewf(double[,] x,
            double[] y,
            double[] w,
            double[] c,
            int n,
            int m,
            int k,
            double diffstep,
            lsfitstate state)
        {
            int i = 0;
            int i_ = 0;

            alglib.ap.assert(n>=1, "LSFitCreateWF: N<1!");
            alglib.ap.assert(m>=1, "LSFitCreateWF: M<1!");
            alglib.ap.assert(k>=1, "LSFitCreateWF: K<1!");
            alglib.ap.assert(alglib.ap.len(c)>=k, "LSFitCreateWF: length(C)<K!");
            alglib.ap.assert(apserv.isfinitevector(c, k), "LSFitCreateWF: C contains infinite or NaN values!");
            alglib.ap.assert(alglib.ap.len(y)>=n, "LSFitCreateWF: length(Y)<N!");
            alglib.ap.assert(apserv.isfinitevector(y, n), "LSFitCreateWF: Y contains infinite or NaN values!");
            alglib.ap.assert(alglib.ap.len(w)>=n, "LSFitCreateWF: length(W)<N!");
            alglib.ap.assert(apserv.isfinitevector(w, n), "LSFitCreateWF: W contains infinite or NaN values!");
            alglib.ap.assert(alglib.ap.rows(x)>=n, "LSFitCreateWF: rows(X)<N!");
            alglib.ap.assert(alglib.ap.cols(x)>=m, "LSFitCreateWF: cols(X)<M!");
            alglib.ap.assert(apserv.apservisfinitematrix(x, n, m), "LSFitCreateWF: X contains infinite or NaN values!");
            alglib.ap.assert(math.isfinite(diffstep), "LSFitCreateWF: DiffStep is not finite!");
            alglib.ap.assert((double)(diffstep)>(double)(0), "LSFitCreateWF: DiffStep<=0!");
            state.npoints = n;
            state.nweights = n;
            state.wkind = 1;
            state.m = m;
            state.k = k;
            lsfitsetcond(state, 0.0, 0.0, 0);
            lsfitsetstpmax(state, 0.0);
            lsfitsetxrep(state, false);
            state.taskx = new double[n, m];
            state.tasky = new double[n];
            state.w = new double[n];
            state.c = new double[k];
            state.x = new double[m];
            for(i_=0; i_<=k-1;i_++)
            {
                state.c[i_] = c[i_];
            }
            for(i_=0; i_<=n-1;i_++)
            {
                state.w[i_] = w[i_];
            }
            for(i=0; i<=n-1; i++)
            {
                for(i_=0; i_<=m-1;i_++)
                {
                    state.taskx[i,i_] = x[i,i_];
                }
                state.tasky[i] = y[i];
            }
            state.s = new double[k];
            state.bndl = new double[k];
            state.bndu = new double[k];
            for(i=0; i<=k-1; i++)
            {
                state.s[i] = 1.0;
                state.bndl[i] = Double.NegativeInfinity;
                state.bndu[i] = Double.PositiveInfinity;
            }
            state.optalgo = 0;
            state.prevnpt = -1;
            state.prevalgo = -1;
            minlm.minlmcreatev(k, n, state.c, diffstep, state.optstate);
            lsfitclearrequestfields(state);
            state.rstate.ia = new int[4+1];
            state.rstate.ra = new double[2+1];
            state.rstate.stage = -1;
        }


        /*************************************************************************
        Nonlinear least squares fitting using function values only.

        Combination of numerical differentiation and secant updates is used to
        obtain function Jacobian.

        Nonlinear task min(F(c)) is solved, where

            F(c) = (f(c,x[0])-y[0])^2 + ... + (f(c,x[n-1])-y[n-1])^2,

            * N is a number of points,
            * M is a dimension of a space points belong to,
            * K is a dimension of a space of parameters being fitted,
            * w is an N-dimensional vector of weight coefficients,
            * x is a set of N points, each of them is an M-dimensional vector,
            * c is a K-dimensional vector of parameters being fitted

        This subroutine uses only f(c,x[i]).

        INPUT PARAMETERS:
            X       -   array[0..N-1,0..M-1], points (one row = one point)
            Y       -   array[0..N-1], function values.
            C       -   array[0..K-1], initial approximation to the solution,
            N       -   number of points, N>1
            M       -   dimension of space
            K       -   number of parameters being fitted
            DiffStep-   numerical differentiation step;
                        should not be very small or large;
                        large = loss of accuracy
                        small = growth of round-off errors

        OUTPUT PARAMETERS:
            State   -   structure which stores algorithm state

          -- ALGLIB --
             Copyright 18.10.2008 by Bochkanov Sergey
        *************************************************************************/
        public static void lsfitcreatef(double[,] x,
            double[] y,
            double[] c,
            int n,
            int m,
            int k,
            double diffstep,
            lsfitstate state)
        {
            int i = 0;
            int i_ = 0;

            alglib.ap.assert(n>=1, "LSFitCreateF: N<1!");
            alglib.ap.assert(m>=1, "LSFitCreateF: M<1!");
            alglib.ap.assert(k>=1, "LSFitCreateF: K<1!");
            alglib.ap.assert(alglib.ap.len(c)>=k, "LSFitCreateF: length(C)<K!");
            alglib.ap.assert(apserv.isfinitevector(c, k), "LSFitCreateF: C contains infinite or NaN values!");
            alglib.ap.assert(alglib.ap.len(y)>=n, "LSFitCreateF: length(Y)<N!");
            alglib.ap.assert(apserv.isfinitevector(y, n), "LSFitCreateF: Y contains infinite or NaN values!");
            alglib.ap.assert(alglib.ap.rows(x)>=n, "LSFitCreateF: rows(X)<N!");
            alglib.ap.assert(alglib.ap.cols(x)>=m, "LSFitCreateF: cols(X)<M!");
            alglib.ap.assert(apserv.apservisfinitematrix(x, n, m), "LSFitCreateF: X contains infinite or NaN values!");
            alglib.ap.assert(alglib.ap.rows(x)>=n, "LSFitCreateF: rows(X)<N!");
            alglib.ap.assert(alglib.ap.cols(x)>=m, "LSFitCreateF: cols(X)<M!");
            alglib.ap.assert(apserv.apservisfinitematrix(x, n, m), "LSFitCreateF: X contains infinite or NaN values!");
            alglib.ap.assert(math.isfinite(diffstep), "LSFitCreateF: DiffStep is not finite!");
            alglib.ap.assert((double)(diffstep)>(double)(0), "LSFitCreateF: DiffStep<=0!");
            state.npoints = n;
            state.wkind = 0;
            state.m = m;
            state.k = k;
            lsfitsetcond(state, 0.0, 0.0, 0);
            lsfitsetstpmax(state, 0.0);
            lsfitsetxrep(state, false);
            state.taskx = new double[n, m];
            state.tasky = new double[n];
            state.c = new double[k];
            state.x = new double[m];
            for(i_=0; i_<=k-1;i_++)
            {
                state.c[i_] = c[i_];
            }
            for(i=0; i<=n-1; i++)
            {
                for(i_=0; i_<=m-1;i_++)
                {
                    state.taskx[i,i_] = x[i,i_];
                }
                state.tasky[i] = y[i];
            }
            state.s = new double[k];
            state.bndl = new double[k];
            state.bndu = new double[k];
            for(i=0; i<=k-1; i++)
            {
                state.s[i] = 1.0;
                state.bndl[i] = Double.NegativeInfinity;
                state.bndu[i] = Double.PositiveInfinity;
            }
            state.optalgo = 0;
            state.prevnpt = -1;
            state.prevalgo = -1;
            minlm.minlmcreatev(k, n, state.c, diffstep, state.optstate);
            lsfitclearrequestfields(state);
            state.rstate.ia = new int[4+1];
            state.rstate.ra = new double[2+1];
            state.rstate.stage = -1;
        }


        /*************************************************************************
        Weighted nonlinear least squares fitting using gradient only.

        Nonlinear task min(F(c)) is solved, where

            F(c) = (w[0]*(f(c,x[0])-y[0]))^2 + ... + (w[n-1]*(f(c,x[n-1])-y[n-1]))^2,
            
            * N is a number of points,
            * M is a dimension of a space points belong to,
            * K is a dimension of a space of parameters being fitted,
            * w is an N-dimensional vector of weight coefficients,
            * x is a set of N points, each of them is an M-dimensional vector,
            * c is a K-dimensional vector of parameters being fitted
            
        This subroutine uses only f(c,x[i]) and its gradient.
            
        INPUT PARAMETERS:
            X       -   array[0..N-1,0..M-1], points (one row = one point)
            Y       -   array[0..N-1], function values.
            W       -   weights, array[0..N-1]
            C       -   array[0..K-1], initial approximation to the solution,
            N       -   number of points, N>1
            M       -   dimension of space
            K       -   number of parameters being fitted
            CheapFG -   boolean flag, which is:
                        * True  if both function and gradient calculation complexity
                                are less than O(M^2).  An improved  algorithm  can
                                be  used  which corresponds  to  FGJ  scheme  from
                                MINLM unit.
                        * False otherwise.
                                Standard Jacibian-bases  Levenberg-Marquardt  algo
                                will be used (FJ scheme).

        OUTPUT PARAMETERS:
            State   -   structure which stores algorithm state

        See also:
            LSFitResults
            LSFitCreateFG (fitting without weights)
            LSFitCreateWFGH (fitting using Hessian)
            LSFitCreateFGH (fitting using Hessian, without weights)

          -- ALGLIB --
             Copyright 17.08.2009 by Bochkanov Sergey
        *************************************************************************/
        public static void lsfitcreatewfg(double[,] x,
            double[] y,
            double[] w,
            double[] c,
            int n,
            int m,
            int k,
            bool cheapfg,
            lsfitstate state)
        {
            int i = 0;
            int i_ = 0;

            alglib.ap.assert(n>=1, "LSFitCreateWFG: N<1!");
            alglib.ap.assert(m>=1, "LSFitCreateWFG: M<1!");
            alglib.ap.assert(k>=1, "LSFitCreateWFG: K<1!");
            alglib.ap.assert(alglib.ap.len(c)>=k, "LSFitCreateWFG: length(C)<K!");
            alglib.ap.assert(apserv.isfinitevector(c, k), "LSFitCreateWFG: C contains infinite or NaN values!");
            alglib.ap.assert(alglib.ap.len(y)>=n, "LSFitCreateWFG: length(Y)<N!");
            alglib.ap.assert(apserv.isfinitevector(y, n), "LSFitCreateWFG: Y contains infinite or NaN values!");
            alglib.ap.assert(alglib.ap.len(w)>=n, "LSFitCreateWFG: length(W)<N!");
            alglib.ap.assert(apserv.isfinitevector(w, n), "LSFitCreateWFG: W contains infinite or NaN values!");
            alglib.ap.assert(alglib.ap.rows(x)>=n, "LSFitCreateWFG: rows(X)<N!");
            alglib.ap.assert(alglib.ap.cols(x)>=m, "LSFitCreateWFG: cols(X)<M!");
            alglib.ap.assert(apserv.apservisfinitematrix(x, n, m), "LSFitCreateWFG: X contains infinite or NaN values!");
            state.npoints = n;
            state.nweights = n;
            state.wkind = 1;
            state.m = m;
            state.k = k;
            lsfitsetcond(state, 0.0, 0.0, 0);
            lsfitsetstpmax(state, 0.0);
            lsfitsetxrep(state, false);
            state.taskx = new double[n, m];
            state.tasky = new double[n];
            state.w = new double[n];
            state.c = new double[k];
            state.x = new double[m];
            state.g = new double[k];
            for(i_=0; i_<=k-1;i_++)
            {
                state.c[i_] = c[i_];
            }
            for(i_=0; i_<=n-1;i_++)
            {
                state.w[i_] = w[i_];
            }
            for(i=0; i<=n-1; i++)
            {
                for(i_=0; i_<=m-1;i_++)
                {
                    state.taskx[i,i_] = x[i,i_];
                }
                state.tasky[i] = y[i];
            }
            state.s = new double[k];
            state.bndl = new double[k];
            state.bndu = new double[k];
            for(i=0; i<=k-1; i++)
            {
                state.s[i] = 1.0;
                state.bndl[i] = Double.NegativeInfinity;
                state.bndu[i] = Double.PositiveInfinity;
            }
            state.optalgo = 1;
            state.prevnpt = -1;
            state.prevalgo = -1;
            if( cheapfg )
            {
                minlm.minlmcreatevgj(k, n, state.c, state.optstate);
            }
            else
            {
                minlm.minlmcreatevj(k, n, state.c, state.optstate);
            }
            lsfitclearrequestfields(state);
            state.rstate.ia = new int[4+1];
            state.rstate.ra = new double[2+1];
            state.rstate.stage = -1;
        }


        /*************************************************************************
        Nonlinear least squares fitting using gradient only, without individual
        weights.

        Nonlinear task min(F(c)) is solved, where

            F(c) = ((f(c,x[0])-y[0]))^2 + ... + ((f(c,x[n-1])-y[n-1]))^2,

            * N is a number of points,
            * M is a dimension of a space points belong to,
            * K is a dimension of a space of parameters being fitted,
            * x is a set of N points, each of them is an M-dimensional vector,
            * c is a K-dimensional vector of parameters being fitted

        This subroutine uses only f(c,x[i]) and its gradient.

        INPUT PARAMETERS:
            X       -   array[0..N-1,0..M-1], points (one row = one point)
            Y       -   array[0..N-1], function values.
            C       -   array[0..K-1], initial approximation to the solution,
            N       -   number of points, N>1
            M       -   dimension of space
            K       -   number of parameters being fitted
            CheapFG -   boolean flag, which is:
                        * True  if both function and gradient calculation complexity
                                are less than O(M^2).  An improved  algorithm  can
                                be  used  which corresponds  to  FGJ  scheme  from
                                MINLM unit.
                        * False otherwise.
                                Standard Jacibian-bases  Levenberg-Marquardt  algo
                                will be used (FJ scheme).

        OUTPUT PARAMETERS:
            State   -   structure which stores algorithm state

          -- ALGLIB --
             Copyright 17.08.2009 by Bochkanov Sergey
        *************************************************************************/
        public static void lsfitcreatefg(double[,] x,
            double[] y,
            double[] c,
            int n,
            int m,
            int k,
            bool cheapfg,
            lsfitstate state)
        {
            int i = 0;
            int i_ = 0;

            alglib.ap.assert(n>=1, "LSFitCreateFG: N<1!");
            alglib.ap.assert(m>=1, "LSFitCreateFG: M<1!");
            alglib.ap.assert(k>=1, "LSFitCreateFG: K<1!");
            alglib.ap.assert(alglib.ap.len(c)>=k, "LSFitCreateFG: length(C)<K!");
            alglib.ap.assert(apserv.isfinitevector(c, k), "LSFitCreateFG: C contains infinite or NaN values!");
            alglib.ap.assert(alglib.ap.len(y)>=n, "LSFitCreateFG: length(Y)<N!");
            alglib.ap.assert(apserv.isfinitevector(y, n), "LSFitCreateFG: Y contains infinite or NaN values!");
            alglib.ap.assert(alglib.ap.rows(x)>=n, "LSFitCreateFG: rows(X)<N!");
            alglib.ap.assert(alglib.ap.cols(x)>=m, "LSFitCreateFG: cols(X)<M!");
            alglib.ap.assert(apserv.apservisfinitematrix(x, n, m), "LSFitCreateFG: X contains infinite or NaN values!");
            alglib.ap.assert(alglib.ap.rows(x)>=n, "LSFitCreateFG: rows(X)<N!");
            alglib.ap.assert(alglib.ap.cols(x)>=m, "LSFitCreateFG: cols(X)<M!");
            alglib.ap.assert(apserv.apservisfinitematrix(x, n, m), "LSFitCreateFG: X contains infinite or NaN values!");
            state.npoints = n;
            state.wkind = 0;
            state.m = m;
            state.k = k;
            lsfitsetcond(state, 0.0, 0.0, 0);
            lsfitsetstpmax(state, 0.0);
            lsfitsetxrep(state, false);
            state.taskx = new double[n, m];
            state.tasky = new double[n];
            state.c = new double[k];
            state.x = new double[m];
            state.g = new double[k];
            for(i_=0; i_<=k-1;i_++)
            {
                state.c[i_] = c[i_];
            }
            for(i=0; i<=n-1; i++)
            {
                for(i_=0; i_<=m-1;i_++)
                {
                    state.taskx[i,i_] = x[i,i_];
                }
                state.tasky[i] = y[i];
            }
            state.s = new double[k];
            state.bndl = new double[k];
            state.bndu = new double[k];
            for(i=0; i<=k-1; i++)
            {
                state.s[i] = 1.0;
                state.bndl[i] = Double.NegativeInfinity;
                state.bndu[i] = Double.PositiveInfinity;
            }
            state.optalgo = 1;
            state.prevnpt = -1;
            state.prevalgo = -1;
            if( cheapfg )
            {
                minlm.minlmcreatevgj(k, n, state.c, state.optstate);
            }
            else
            {
                minlm.minlmcreatevj(k, n, state.c, state.optstate);
            }
            lsfitclearrequestfields(state);
            state.rstate.ia = new int[4+1];
            state.rstate.ra = new double[2+1];
            state.rstate.stage = -1;
        }


        /*************************************************************************
        Weighted nonlinear least squares fitting using gradient/Hessian.

        Nonlinear task min(F(c)) is solved, where

            F(c) = (w[0]*(f(c,x[0])-y[0]))^2 + ... + (w[n-1]*(f(c,x[n-1])-y[n-1]))^2,

            * N is a number of points,
            * M is a dimension of a space points belong to,
            * K is a dimension of a space of parameters being fitted,
            * w is an N-dimensional vector of weight coefficients,
            * x is a set of N points, each of them is an M-dimensional vector,
            * c is a K-dimensional vector of parameters being fitted

        This subroutine uses f(c,x[i]), its gradient and its Hessian.

        INPUT PARAMETERS:
            X       -   array[0..N-1,0..M-1], points (one row = one point)
            Y       -   array[0..N-1], function values.
            W       -   weights, array[0..N-1]
            C       -   array[0..K-1], initial approximation to the solution,
            N       -   number of points, N>1
            M       -   dimension of space
            K       -   number of parameters being fitted

        OUTPUT PARAMETERS:
            State   -   structure which stores algorithm state

          -- ALGLIB --
             Copyright 17.08.2009 by Bochkanov Sergey
        *************************************************************************/
        public static void lsfitcreatewfgh(double[,] x,
            double[] y,
            double[] w,
            double[] c,
            int n,
            int m,
            int k,
            lsfitstate state)
        {
            int i = 0;
            int i_ = 0;

            alglib.ap.assert(n>=1, "LSFitCreateWFGH: N<1!");
            alglib.ap.assert(m>=1, "LSFitCreateWFGH: M<1!");
            alglib.ap.assert(k>=1, "LSFitCreateWFGH: K<1!");
            alglib.ap.assert(alglib.ap.len(c)>=k, "LSFitCreateWFGH: length(C)<K!");
            alglib.ap.assert(apserv.isfinitevector(c, k), "LSFitCreateWFGH: C contains infinite or NaN values!");
            alglib.ap.assert(alglib.ap.len(y)>=n, "LSFitCreateWFGH: length(Y)<N!");
            alglib.ap.assert(apserv.isfinitevector(y, n), "LSFitCreateWFGH: Y contains infinite or NaN values!");
            alglib.ap.assert(alglib.ap.len(w)>=n, "LSFitCreateWFGH: length(W)<N!");
            alglib.ap.assert(apserv.isfinitevector(w, n), "LSFitCreateWFGH: W contains infinite or NaN values!");
            alglib.ap.assert(alglib.ap.rows(x)>=n, "LSFitCreateWFGH: rows(X)<N!");
            alglib.ap.assert(alglib.ap.cols(x)>=m, "LSFitCreateWFGH: cols(X)<M!");
            alglib.ap.assert(apserv.apservisfinitematrix(x, n, m), "LSFitCreateWFGH: X contains infinite or NaN values!");
            state.npoints = n;
            state.nweights = n;
            state.wkind = 1;
            state.m = m;
            state.k = k;
            lsfitsetcond(state, 0.0, 0.0, 0);
            lsfitsetstpmax(state, 0.0);
            lsfitsetxrep(state, false);
            state.taskx = new double[n, m];
            state.tasky = new double[n];
            state.w = new double[n];
            state.c = new double[k];
            state.h = new double[k, k];
            state.x = new double[m];
            state.g = new double[k];
            for(i_=0; i_<=k-1;i_++)
            {
                state.c[i_] = c[i_];
            }
            for(i_=0; i_<=n-1;i_++)
            {
                state.w[i_] = w[i_];
            }
            for(i=0; i<=n-1; i++)
            {
                for(i_=0; i_<=m-1;i_++)
                {
                    state.taskx[i,i_] = x[i,i_];
                }
                state.tasky[i] = y[i];
            }
            state.s = new double[k];
            state.bndl = new double[k];
            state.bndu = new double[k];
            for(i=0; i<=k-1; i++)
            {
                state.s[i] = 1.0;
                state.bndl[i] = Double.NegativeInfinity;
                state.bndu[i] = Double.PositiveInfinity;
            }
            state.optalgo = 2;
            state.prevnpt = -1;
            state.prevalgo = -1;
            minlm.minlmcreatefgh(k, state.c, state.optstate);
            lsfitclearrequestfields(state);
            state.rstate.ia = new int[4+1];
            state.rstate.ra = new double[2+1];
            state.rstate.stage = -1;
        }


        /*************************************************************************
        Nonlinear least squares fitting using gradient/Hessian, without individial
        weights.

        Nonlinear task min(F(c)) is solved, where

            F(c) = ((f(c,x[0])-y[0]))^2 + ... + ((f(c,x[n-1])-y[n-1]))^2,

            * N is a number of points,
            * M is a dimension of a space points belong to,
            * K is a dimension of a space of parameters being fitted,
            * x is a set of N points, each of them is an M-dimensional vector,
            * c is a K-dimensional vector of parameters being fitted

        This subroutine uses f(c,x[i]), its gradient and its Hessian.

        INPUT PARAMETERS:
            X       -   array[0..N-1,0..M-1], points (one row = one point)
            Y       -   array[0..N-1], function values.
            C       -   array[0..K-1], initial approximation to the solution,
            N       -   number of points, N>1
            M       -   dimension of space
            K       -   number of parameters being fitted

        OUTPUT PARAMETERS:
            State   -   structure which stores algorithm state


          -- ALGLIB --
             Copyright 17.08.2009 by Bochkanov Sergey
        *************************************************************************/
        public static void lsfitcreatefgh(double[,] x,
            double[] y,
            double[] c,
            int n,
            int m,
            int k,
            lsfitstate state)
        {
            int i = 0;
            int i_ = 0;

            alglib.ap.assert(n>=1, "LSFitCreateFGH: N<1!");
            alglib.ap.assert(m>=1, "LSFitCreateFGH: M<1!");
            alglib.ap.assert(k>=1, "LSFitCreateFGH: K<1!");
            alglib.ap.assert(alglib.ap.len(c)>=k, "LSFitCreateFGH: length(C)<K!");
            alglib.ap.assert(apserv.isfinitevector(c, k), "LSFitCreateFGH: C contains infinite or NaN values!");
            alglib.ap.assert(alglib.ap.len(y)>=n, "LSFitCreateFGH: length(Y)<N!");
            alglib.ap.assert(apserv.isfinitevector(y, n), "LSFitCreateFGH: Y contains infinite or NaN values!");
            alglib.ap.assert(alglib.ap.rows(x)>=n, "LSFitCreateFGH: rows(X)<N!");
            alglib.ap.assert(alglib.ap.cols(x)>=m, "LSFitCreateFGH: cols(X)<M!");
            alglib.ap.assert(apserv.apservisfinitematrix(x, n, m), "LSFitCreateFGH: X contains infinite or NaN values!");
            state.npoints = n;
            state.wkind = 0;
            state.m = m;
            state.k = k;
            lsfitsetcond(state, 0.0, 0.0, 0);
            lsfitsetstpmax(state, 0.0);
            lsfitsetxrep(state, false);
            state.taskx = new double[n, m];
            state.tasky = new double[n];
            state.c = new double[k];
            state.h = new double[k, k];
            state.x = new double[m];
            state.g = new double[k];
            for(i_=0; i_<=k-1;i_++)
            {
                state.c[i_] = c[i_];
            }
            for(i=0; i<=n-1; i++)
            {
                for(i_=0; i_<=m-1;i_++)
                {
                    state.taskx[i,i_] = x[i,i_];
                }
                state.tasky[i] = y[i];
            }
            state.s = new double[k];
            state.bndl = new double[k];
            state.bndu = new double[k];
            for(i=0; i<=k-1; i++)
            {
                state.s[i] = 1.0;
                state.bndl[i] = Double.NegativeInfinity;
                state.bndu[i] = Double.PositiveInfinity;
            }
            state.optalgo = 2;
            state.prevnpt = -1;
            state.prevalgo = -1;
            minlm.minlmcreatefgh(k, state.c, state.optstate);
            lsfitclearrequestfields(state);
            state.rstate.ia = new int[4+1];
            state.rstate.ra = new double[2+1];
            state.rstate.stage = -1;
        }


        /*************************************************************************
        Stopping conditions for nonlinear least squares fitting.

        INPUT PARAMETERS:
            State   -   structure which stores algorithm state
            EpsF    -   stopping criterion. Algorithm stops if
                        |F(k+1)-F(k)| <= EpsF*max{|F(k)|, |F(k+1)|, 1}
            EpsX    -   >=0
                        The subroutine finishes its work if  on  k+1-th  iteration
                        the condition |v|<=EpsX is fulfilled, where:
                        * |.| means Euclidian norm
                        * v - scaled step vector, v[i]=dx[i]/s[i]
                        * dx - ste pvector, dx=X(k+1)-X(k)
                        * s - scaling coefficients set by LSFitSetScale()
            MaxIts  -   maximum number of iterations. If MaxIts=0, the  number  of
                        iterations   is    unlimited.   Only   Levenberg-Marquardt
                        iterations  are  counted  (L-BFGS/CG  iterations  are  NOT
                        counted because their cost is very low compared to that of
                        LM).

        NOTE

        Passing EpsF=0, EpsX=0 and MaxIts=0 (simultaneously) will lead to automatic
        stopping criterion selection (according to the scheme used by MINLM unit).


          -- ALGLIB --
             Copyright 17.08.2009 by Bochkanov Sergey
        *************************************************************************/
        public static void lsfitsetcond(lsfitstate state,
            double epsf,
            double epsx,
            int maxits)
        {
            alglib.ap.assert(math.isfinite(epsf), "LSFitSetCond: EpsF is not finite!");
            alglib.ap.assert((double)(epsf)>=(double)(0), "LSFitSetCond: negative EpsF!");
            alglib.ap.assert(math.isfinite(epsx), "LSFitSetCond: EpsX is not finite!");
            alglib.ap.assert((double)(epsx)>=(double)(0), "LSFitSetCond: negative EpsX!");
            alglib.ap.assert(maxits>=0, "LSFitSetCond: negative MaxIts!");
            state.epsf = epsf;
            state.epsx = epsx;
            state.maxits = maxits;
        }


        /*************************************************************************
        This function sets maximum step length

        INPUT PARAMETERS:
            State   -   structure which stores algorithm state
            StpMax  -   maximum step length, >=0. Set StpMax to 0.0,  if you don't
                        want to limit step length.

        Use this subroutine when you optimize target function which contains exp()
        or  other  fast  growing  functions,  and optimization algorithm makes too
        large  steps  which  leads  to overflow. This function allows us to reject
        steps  that  are  too  large  (and  therefore  expose  us  to the possible
        overflow) without actually calculating function value at the x+stp*d.

        NOTE: non-zero StpMax leads to moderate  performance  degradation  because
        intermediate  step  of  preconditioned L-BFGS optimization is incompatible
        with limits on step size.

          -- ALGLIB --
             Copyright 02.04.2010 by Bochkanov Sergey
        *************************************************************************/
        public static void lsfitsetstpmax(lsfitstate state,
            double stpmax)
        {
            alglib.ap.assert((double)(stpmax)>=(double)(0), "LSFitSetStpMax: StpMax<0!");
            state.stpmax = stpmax;
        }


        /*************************************************************************
        This function turns on/off reporting.

        INPUT PARAMETERS:
            State   -   structure which stores algorithm state
            NeedXRep-   whether iteration reports are needed or not
            
        When reports are needed, State.C (current parameters) and State.F (current
        value of fitting function) are reported.


          -- ALGLIB --
             Copyright 15.08.2010 by Bochkanov Sergey
        *************************************************************************/
        public static void lsfitsetxrep(lsfitstate state,
            bool needxrep)
        {
            state.xrep = needxrep;
        }


        /*************************************************************************
        This function sets scaling coefficients for underlying optimizer.

        ALGLIB optimizers use scaling matrices to test stopping  conditions  (step
        size and gradient are scaled before comparison with tolerances).  Scale of
        the I-th variable is a translation invariant measure of:
        a) "how large" the variable is
        b) how large the step should be to make significant changes in the function

        Generally, scale is NOT considered to be a form of preconditioner.  But LM
        optimizer is unique in that it uses scaling matrix both  in  the  stopping
        condition tests and as Marquardt damping factor.

        Proper scaling is very important for the algorithm performance. It is less
        important for the quality of results, but still has some influence (it  is
        easier  to  converge  when  variables  are  properly  scaled, so premature
        stopping is possible when very badly scalled variables are  combined  with
        relaxed stopping conditions).

        INPUT PARAMETERS:
            State   -   structure stores algorithm state
            S       -   array[N], non-zero scaling coefficients
                        S[i] may be negative, sign doesn't matter.

          -- ALGLIB --
             Copyright 14.01.2011 by Bochkanov Sergey
        *************************************************************************/
        public static void lsfitsetscale(lsfitstate state,
            double[] s)
        {
            int i = 0;

            alglib.ap.assert(alglib.ap.len(s)>=state.k, "LSFitSetScale: Length(S)<K");
            for(i=0; i<=state.k-1; i++)
            {
                alglib.ap.assert(math.isfinite(s[i]), "LSFitSetScale: S contains infinite or NAN elements");
                alglib.ap.assert((double)(s[i])!=(double)(0), "LSFitSetScale: S contains infinite or NAN elements");
                state.s[i] = s[i];
            }
        }


        /*************************************************************************
        This function sets boundary constraints for underlying optimizer

        Boundary constraints are inactive by default (after initial creation).
        They are preserved until explicitly turned off with another SetBC() call.

        INPUT PARAMETERS:
            State   -   structure stores algorithm state
            BndL    -   lower bounds, array[K].
                        If some (all) variables are unbounded, you may specify
                        very small number or -INF (latter is recommended because
                        it will allow solver to use better algorithm).
            BndU    -   upper bounds, array[K].
                        If some (all) variables are unbounded, you may specify
                        very large number or +INF (latter is recommended because
                        it will allow solver to use better algorithm).

        NOTE 1: it is possible to specify BndL[i]=BndU[i]. In this case I-th
        variable will be "frozen" at X[i]=BndL[i]=BndU[i].

        NOTE 2: unlike other constrained optimization algorithms, this solver  has
        following useful properties:
        * bound constraints are always satisfied exactly
        * function is evaluated only INSIDE area specified by bound constraints

          -- ALGLIB --
             Copyright 14.01.2011 by Bochkanov Sergey
        *************************************************************************/
        public static void lsfitsetbc(lsfitstate state,
            double[] bndl,
            double[] bndu)
        {
            int i = 0;
            int k = 0;

            k = state.k;
            alglib.ap.assert(alglib.ap.len(bndl)>=k, "LSFitSetBC: Length(BndL)<K");
            alglib.ap.assert(alglib.ap.len(bndu)>=k, "LSFitSetBC: Length(BndU)<K");
            for(i=0; i<=k-1; i++)
            {
                alglib.ap.assert(math.isfinite(bndl[i]) || Double.IsNegativeInfinity(bndl[i]), "LSFitSetBC: BndL contains NAN or +INF");
                alglib.ap.assert(math.isfinite(bndu[i]) || Double.IsPositiveInfinity(bndu[i]), "LSFitSetBC: BndU contains NAN or -INF");
                if( math.isfinite(bndl[i]) && math.isfinite(bndu[i]) )
                {
                    alglib.ap.assert((double)(bndl[i])<=(double)(bndu[i]), "LSFitSetBC: BndL[i]>BndU[i]");
                }
                state.bndl[i] = bndl[i];
                state.bndu[i] = bndu[i];
            }
        }


        /*************************************************************************
        NOTES:

        1. this algorithm is somewhat unusual because it works with  parameterized
           function f(C,X), where X is a function argument (we  have  many  points
           which are characterized by different  argument  values),  and  C  is  a
           parameter to fit.

           For example, if we want to do linear fit by f(c0,c1,x) = c0*x+c1,  then
           x will be argument, and {c0,c1} will be parameters.
           
           It is important to understand that this algorithm finds minimum in  the
           space of function PARAMETERS (not arguments), so it  needs  derivatives
           of f() with respect to C, not X.
           
           In the example above it will need f=c0*x+c1 and {df/dc0,df/dc1} = {x,1}
           instead of {df/dx} = {c0}.

        2. Callback functions accept C as the first parameter, and X as the second

        3. If  state  was  created  with  LSFitCreateFG(),  algorithm  needs  just
           function   and   its   gradient,   but   if   state   was  created with
           LSFitCreateFGH(), algorithm will need function, gradient and Hessian.
           
           According  to  the  said  above,  there  ase  several  versions of this
           function, which accept different sets of callbacks.
           
           This flexibility opens way to subtle errors - you may create state with
           LSFitCreateFGH() (optimization using Hessian), but call function  which
           does not accept Hessian. So when algorithm will request Hessian,  there
           will be no callback to call. In this case exception will be thrown.
           
           Be careful to avoid such errors because there is no way to find them at
           compile time - you can see them at runtime only.

          -- ALGLIB --
             Copyright 17.08.2009 by Bochkanov Sergey
        *************************************************************************/
        public static bool lsfititeration(lsfitstate state)
        {
            bool result = new bool();
            int n = 0;
            int m = 0;
            int k = 0;
            int i = 0;
            int j = 0;
            double v = 0;
            double vv = 0;
            double relcnt = 0;
            int i_ = 0;

            
            //
            // Reverse communication preparations
            // I know it looks ugly, but it works the same way
            // anywhere from C++ to Python.
            //
            // This code initializes locals by:
            // * random values determined during code
            //   generation - on first subroutine call
            // * values from previous call - on subsequent calls
            //
            if( state.rstate.stage>=0 )
            {
                n = state.rstate.ia[0];
                m = state.rstate.ia[1];
                k = state.rstate.ia[2];
                i = state.rstate.ia[3];
                j = state.rstate.ia[4];
                v = state.rstate.ra[0];
                vv = state.rstate.ra[1];
                relcnt = state.rstate.ra[2];
            }
            else
            {
                n = -983;
                m = -989;
                k = -834;
                i = 900;
                j = -287;
                v = 364;
                vv = 214;
                relcnt = -338;
            }
            if( state.rstate.stage==0 )
            {
                goto lbl_0;
            }
            if( state.rstate.stage==1 )
            {
                goto lbl_1;
            }
            if( state.rstate.stage==2 )
            {
                goto lbl_2;
            }
            if( state.rstate.stage==3 )
            {
                goto lbl_3;
            }
            if( state.rstate.stage==4 )
            {
                goto lbl_4;
            }
            if( state.rstate.stage==5 )
            {
                goto lbl_5;
            }
            if( state.rstate.stage==6 )
            {
                goto lbl_6;
            }
            
            //
            // Routine body
            //
            
            //
            // init
            //
            if( state.wkind==1 )
            {
                alglib.ap.assert(state.npoints==state.nweights, "LSFitFit: number of points is not equal to the number of weights");
            }
            n = state.npoints;
            m = state.m;
            k = state.k;
            minlm.minlmsetcond(state.optstate, 0.0, state.epsf, state.epsx, state.maxits);
            minlm.minlmsetstpmax(state.optstate, state.stpmax);
            minlm.minlmsetxrep(state.optstate, state.xrep);
            minlm.minlmsetscale(state.optstate, state.s);
            minlm.minlmsetbc(state.optstate, state.bndl, state.bndu);
            
            //
            // Optimize
            //
        lbl_7:
            if( !minlm.minlmiteration(state.optstate) )
            {
                goto lbl_8;
            }
            if( !state.optstate.needfi )
            {
                goto lbl_9;
            }
            
            //
            // calculate f[] = wi*(f(xi,c)-yi)
            //
            i = 0;
        lbl_11:
            if( i>n-1 )
            {
                goto lbl_13;
            }
            for(i_=0; i_<=k-1;i_++)
            {
                state.c[i_] = state.optstate.x[i_];
            }
            for(i_=0; i_<=m-1;i_++)
            {
                state.x[i_] = state.taskx[i,i_];
            }
            state.pointindex = i;
            lsfitclearrequestfields(state);
            state.needf = true;
            state.rstate.stage = 0;
            goto lbl_rcomm;
        lbl_0:
            state.needf = false;
            if( state.wkind==1 )
            {
                vv = state.w[i];
            }
            else
            {
                vv = 1.0;
            }
            state.optstate.fi[i] = vv*(state.f-state.tasky[i]);
            i = i+1;
            goto lbl_11;
        lbl_13:
            goto lbl_7;
        lbl_9:
            if( !state.optstate.needf )
            {
                goto lbl_14;
            }
            
            //
            // calculate F = sum (wi*(f(xi,c)-yi))^2
            //
            state.optstate.f = 0;
            i = 0;
        lbl_16:
            if( i>n-1 )
            {
                goto lbl_18;
            }
            for(i_=0; i_<=k-1;i_++)
            {
                state.c[i_] = state.optstate.x[i_];
            }
            for(i_=0; i_<=m-1;i_++)
            {
                state.x[i_] = state.taskx[i,i_];
            }
            state.pointindex = i;
            lsfitclearrequestfields(state);
            state.needf = true;
            state.rstate.stage = 1;
            goto lbl_rcomm;
        lbl_1:
            state.needf = false;
            if( state.wkind==1 )
            {
                vv = state.w[i];
            }
            else
            {
                vv = 1.0;
            }
            state.optstate.f = state.optstate.f+math.sqr(vv*(state.f-state.tasky[i]));
            i = i+1;
            goto lbl_16;
        lbl_18:
            goto lbl_7;
        lbl_14:
            if( !state.optstate.needfg )
            {
                goto lbl_19;
            }
            
            //
            // calculate F/gradF
            //
            state.optstate.f = 0;
            for(i=0; i<=k-1; i++)
            {
                state.optstate.g[i] = 0;
            }
            i = 0;
        lbl_21:
            if( i>n-1 )
            {
                goto lbl_23;
            }
            for(i_=0; i_<=k-1;i_++)
            {
                state.c[i_] = state.optstate.x[i_];
            }
            for(i_=0; i_<=m-1;i_++)
            {
                state.x[i_] = state.taskx[i,i_];
            }
            state.pointindex = i;
            lsfitclearrequestfields(state);
            state.needfg = true;
            state.rstate.stage = 2;
            goto lbl_rcomm;
        lbl_2:
            state.needfg = false;
            if( state.wkind==1 )
            {
                vv = state.w[i];
            }
            else
            {
                vv = 1.0;
            }
            state.optstate.f = state.optstate.f+math.sqr(vv*(state.f-state.tasky[i]));
            v = math.sqr(vv)*2*(state.f-state.tasky[i]);
            for(i_=0; i_<=k-1;i_++)
            {
                state.optstate.g[i_] = state.optstate.g[i_] + v*state.g[i_];
            }
            i = i+1;
            goto lbl_21;
        lbl_23:
            goto lbl_7;
        lbl_19:
            if( !state.optstate.needfij )
            {
                goto lbl_24;
            }
            
            //
            // calculate Fi/jac(Fi)
            //
            i = 0;
        lbl_26:
            if( i>n-1 )
            {
                goto lbl_28;
            }
            for(i_=0; i_<=k-1;i_++)
            {
                state.c[i_] = state.optstate.x[i_];
            }
            for(i_=0; i_<=m-1;i_++)
            {
                state.x[i_] = state.taskx[i,i_];
            }
            state.pointindex = i;
            lsfitclearrequestfields(state);
            state.needfg = true;
            state.rstate.stage = 3;
            goto lbl_rcomm;
        lbl_3:
            state.needfg = false;
            if( state.wkind==1 )
            {
                vv = state.w[i];
            }
            else
            {
                vv = 1.0;
            }
            state.optstate.fi[i] = vv*(state.f-state.tasky[i]);
            for(i_=0; i_<=k-1;i_++)
            {
                state.optstate.j[i,i_] = vv*state.g[i_];
            }
            i = i+1;
            goto lbl_26;
        lbl_28:
            goto lbl_7;
        lbl_24:
            if( !state.optstate.needfgh )
            {
                goto lbl_29;
            }
            
            //
            // calculate F/grad(F)/hess(F)
            //
            state.optstate.f = 0;
            for(i=0; i<=k-1; i++)
            {
                state.optstate.g[i] = 0;
            }
            for(i=0; i<=k-1; i++)
            {
                for(j=0; j<=k-1; j++)
                {
                    state.optstate.h[i,j] = 0;
                }
            }
            i = 0;
        lbl_31:
            if( i>n-1 )
            {
                goto lbl_33;
            }
            for(i_=0; i_<=k-1;i_++)
            {
                state.c[i_] = state.optstate.x[i_];
            }
            for(i_=0; i_<=m-1;i_++)
            {
                state.x[i_] = state.taskx[i,i_];
            }
            state.pointindex = i;
            lsfitclearrequestfields(state);
            state.needfgh = true;
            state.rstate.stage = 4;
            goto lbl_rcomm;
        lbl_4:
            state.needfgh = false;
            if( state.wkind==1 )
            {
                vv = state.w[i];
            }
            else
            {
                vv = 1.0;
            }
            state.optstate.f = state.optstate.f+math.sqr(vv*(state.f-state.tasky[i]));
            v = math.sqr(vv)*2*(state.f-state.tasky[i]);
            for(i_=0; i_<=k-1;i_++)
            {
                state.optstate.g[i_] = state.optstate.g[i_] + v*state.g[i_];
            }
            for(j=0; j<=k-1; j++)
            {
                v = 2*math.sqr(vv)*state.g[j];
                for(i_=0; i_<=k-1;i_++)
                {
                    state.optstate.h[j,i_] = state.optstate.h[j,i_] + v*state.g[i_];
                }
                v = 2*math.sqr(vv)*(state.f-state.tasky[i]);
                for(i_=0; i_<=k-1;i_++)
                {
                    state.optstate.h[j,i_] = state.optstate.h[j,i_] + v*state.h[j,i_];
                }
            }
            i = i+1;
            goto lbl_31;
        lbl_33:
            goto lbl_7;
        lbl_29:
            if( !state.optstate.xupdated )
            {
                goto lbl_34;
            }
            
            //
            // Report new iteration
            //
            for(i_=0; i_<=k-1;i_++)
            {
                state.c[i_] = state.optstate.x[i_];
            }
            state.f = state.optstate.f;
            lsfitclearrequestfields(state);
            state.xupdated = true;
            state.rstate.stage = 5;
            goto lbl_rcomm;
        lbl_5:
            state.xupdated = false;
            goto lbl_7;
        lbl_34:
            goto lbl_7;
        lbl_8:
            minlm.minlmresults(state.optstate, ref state.c, state.optrep);
            state.repterminationtype = state.optrep.terminationtype;
            state.repiterationscount = state.optrep.iterationscount;
            
            //
            // calculate errors
            //
            if( state.repterminationtype<=0 )
            {
                goto lbl_36;
            }
            state.reprmserror = 0;
            state.repwrmserror = 0;
            state.repavgerror = 0;
            state.repavgrelerror = 0;
            state.repmaxerror = 0;
            relcnt = 0;
            i = 0;
        lbl_38:
            if( i>n-1 )
            {
                goto lbl_40;
            }
            for(i_=0; i_<=k-1;i_++)
            {
                state.c[i_] = state.c[i_];
            }
            for(i_=0; i_<=m-1;i_++)
            {
                state.x[i_] = state.taskx[i,i_];
            }
            state.pointindex = i;
            lsfitclearrequestfields(state);
            state.needf = true;
            state.rstate.stage = 6;
            goto lbl_rcomm;
        lbl_6:
            state.needf = false;
            v = state.f;
            if( state.wkind==1 )
            {
                vv = state.w[i];
            }
            else
            {
                vv = 1.0;
            }
            state.reprmserror = state.reprmserror+math.sqr(v-state.tasky[i]);
            state.repwrmserror = state.repwrmserror+math.sqr(vv*(v-state.tasky[i]));
            state.repavgerror = state.repavgerror+Math.Abs(v-state.tasky[i]);
            if( (double)(state.tasky[i])!=(double)(0) )
            {
                state.repavgrelerror = state.repavgrelerror+Math.Abs(v-state.tasky[i])/Math.Abs(state.tasky[i]);
                relcnt = relcnt+1;
            }
            state.repmaxerror = Math.Max(state.repmaxerror, Math.Abs(v-state.tasky[i]));
            i = i+1;
            goto lbl_38;
        lbl_40:
            state.reprmserror = Math.Sqrt(state.reprmserror/n);
            state.repwrmserror = Math.Sqrt(state.repwrmserror/n);
            state.repavgerror = state.repavgerror/n;
            if( (double)(relcnt)!=(double)(0) )
            {
                state.repavgrelerror = state.repavgrelerror/relcnt;
            }
        lbl_36:
            result = false;
            return result;
            
            //
            // Saving state
            //
        lbl_rcomm:
            result = true;
            state.rstate.ia[0] = n;
            state.rstate.ia[1] = m;
            state.rstate.ia[2] = k;
            state.rstate.ia[3] = i;
            state.rstate.ia[4] = j;
            state.rstate.ra[0] = v;
            state.rstate.ra[1] = vv;
            state.rstate.ra[2] = relcnt;
            return result;
        }


        /*************************************************************************
        Nonlinear least squares fitting results.

        Called after return from LSFitFit().

        INPUT PARAMETERS:
            State   -   algorithm state

        OUTPUT PARAMETERS:
            Info    -   completetion code:
                            *  1    relative function improvement is no more than
                                    EpsF.
                            *  2    relative step is no more than EpsX.
                            *  4    gradient norm is no more than EpsG
                            *  5    MaxIts steps was taken
                            *  7    stopping conditions are too stringent,
                                    further improvement is impossible
            C       -   array[0..K-1], solution
            Rep     -   optimization report. Following fields are set:
                        * Rep.TerminationType completetion code:
                        * RMSError          rms error on the (X,Y).
                        * AvgError          average error on the (X,Y).
                        * AvgRelError       average relative error on the non-zero Y
                        * MaxError          maximum error
                                            NON-WEIGHTED ERRORS ARE CALCULATED
                        * WRMSError         weighted rms error on the (X,Y).


          -- ALGLIB --
             Copyright 17.08.2009 by Bochkanov Sergey
        *************************************************************************/
        public static void lsfitresults(lsfitstate state,
            ref int info,
            ref double[] c,
            lsfitreport rep)
        {
            int i_ = 0;

            info = 0;
            c = new double[0];

            info = state.repterminationtype;
            if( info>0 )
            {
                c = new double[state.k];
                for(i_=0; i_<=state.k-1;i_++)
                {
                    c[i_] = state.c[i_];
                }
                rep.rmserror = state.reprmserror;
                rep.wrmserror = state.repwrmserror;
                rep.avgerror = state.repavgerror;
                rep.avgrelerror = state.repavgrelerror;
                rep.maxerror = state.repmaxerror;
                rep.iterationscount = state.repiterationscount;
            }
        }


        /*************************************************************************
        Internal subroutine: automatic scaling for LLS tasks.
        NEVER CALL IT DIRECTLY!

        Maps abscissas to [-1,1], standartizes ordinates and correspondingly scales
        constraints. It also scales weights so that max(W[i])=1

        Transformations performed:
        * X, XC         [XA,XB] => [-1,+1]
                        transformation makes min(X)=-1, max(X)=+1

        * Y             [SA,SB] => [0,1]
                        transformation makes mean(Y)=0, stddev(Y)=1
                        
        * YC            transformed accordingly to SA, SB, DC[I]

          -- ALGLIB PROJECT --
             Copyright 08.09.2009 by Bochkanov Sergey
        *************************************************************************/
        public static void lsfitscalexy(ref double[] x,
            ref double[] y,
            ref double[] w,
            int n,
            ref double[] xc,
            ref double[] yc,
            int[] dc,
            int k,
            ref double xa,
            ref double xb,
            ref double sa,
            ref double sb,
            ref double[] xoriginal,
            ref double[] yoriginal)
        {
            double xmin = 0;
            double xmax = 0;
            int i = 0;
            double mx = 0;
            int i_ = 0;

            xa = 0;
            xb = 0;
            sa = 0;
            sb = 0;
            xoriginal = new double[0];
            yoriginal = new double[0];

            alglib.ap.assert(n>=1, "LSFitScaleXY: incorrect N");
            alglib.ap.assert(k>=0, "LSFitScaleXY: incorrect K");
            
            //
            // Calculate xmin/xmax.
            // Force xmin<>xmax.
            //
            xmin = x[0];
            xmax = x[0];
            for(i=1; i<=n-1; i++)
            {
                xmin = Math.Min(xmin, x[i]);
                xmax = Math.Max(xmax, x[i]);
            }
            for(i=0; i<=k-1; i++)
            {
                xmin = Math.Min(xmin, xc[i]);
                xmax = Math.Max(xmax, xc[i]);
            }
            if( (double)(xmin)==(double)(xmax) )
            {
                if( (double)(xmin)==(double)(0) )
                {
                    xmin = -1;
                    xmax = 1;
                }
                else
                {
                    if( (double)(xmin)>(double)(0) )
                    {
                        xmin = 0.5*xmin;
                    }
                    else
                    {
                        xmax = 0.5*xmax;
                    }
                }
            }
            
            //
            // Transform abscissas: map [XA,XB] to [0,1]
            //
            // Store old X[] in XOriginal[] (it will be used
            // to calculate relative error).
            //
            xoriginal = new double[n];
            for(i_=0; i_<=n-1;i_++)
            {
                xoriginal[i_] = x[i_];
            }
            xa = xmin;
            xb = xmax;
            for(i=0; i<=n-1; i++)
            {
                x[i] = 2*(x[i]-0.5*(xa+xb))/(xb-xa);
            }
            for(i=0; i<=k-1; i++)
            {
                alglib.ap.assert(dc[i]>=0, "LSFitScaleXY: internal error!");
                xc[i] = 2*(xc[i]-0.5*(xa+xb))/(xb-xa);
                yc[i] = yc[i]*Math.Pow(0.5*(xb-xa), dc[i]);
            }
            
            //
            // Transform function values: map [SA,SB] to [0,1]
            // SA = mean(Y),
            // SB = SA+stddev(Y).
            //
            // Store old Y[] in YOriginal[] (it will be used
            // to calculate relative error).
            //
            yoriginal = new double[n];
            for(i_=0; i_<=n-1;i_++)
            {
                yoriginal[i_] = y[i_];
            }
            sa = 0;
            for(i=0; i<=n-1; i++)
            {
                sa = sa+y[i];
            }
            sa = sa/n;
            sb = 0;
            for(i=0; i<=n-1; i++)
            {
                sb = sb+math.sqr(y[i]-sa);
            }
            sb = Math.Sqrt(sb/n)+sa;
            if( (double)(sb)==(double)(sa) )
            {
                sb = 2*sa;
            }
            if( (double)(sb)==(double)(sa) )
            {
                sb = sa+1;
            }
            for(i=0; i<=n-1; i++)
            {
                y[i] = (y[i]-sa)/(sb-sa);
            }
            for(i=0; i<=k-1; i++)
            {
                if( dc[i]==0 )
                {
                    yc[i] = (yc[i]-sa)/(sb-sa);
                }
                else
                {
                    yc[i] = yc[i]/(sb-sa);
                }
            }
            
            //
            // Scale weights
            //
            mx = 0;
            for(i=0; i<=n-1; i++)
            {
                mx = Math.Max(mx, Math.Abs(w[i]));
            }
            if( (double)(mx)!=(double)(0) )
            {
                for(i=0; i<=n-1; i++)
                {
                    w[i] = w[i]/mx;
                }
            }
        }


        /*************************************************************************
        Internal spline fitting subroutine

          -- ALGLIB PROJECT --
             Copyright 08.09.2009 by Bochkanov Sergey
        *************************************************************************/
        private static void spline1dfitinternal(int st,
            double[] x,
            double[] y,
            double[] w,
            int n,
            double[] xc,
            double[] yc,
            int[] dc,
            int k,
            int m,
            ref int info,
            spline1d.spline1dinterpolant s,
            spline1dfitreport rep)
        {
            double[,] fmatrix = new double[0,0];
            double[,] cmatrix = new double[0,0];
            double[] y2 = new double[0];
            double[] w2 = new double[0];
            double[] sx = new double[0];
            double[] sy = new double[0];
            double[] sd = new double[0];
            double[] tmp = new double[0];
            double[] xoriginal = new double[0];
            double[] yoriginal = new double[0];
            lsfitreport lrep = new lsfitreport();
            double v0 = 0;
            double v1 = 0;
            double v2 = 0;
            double mx = 0;
            spline1d.spline1dinterpolant s2 = new spline1d.spline1dinterpolant();
            int i = 0;
            int j = 0;
            int relcnt = 0;
            double xa = 0;
            double xb = 0;
            double sa = 0;
            double sb = 0;
            double bl = 0;
            double br = 0;
            double decay = 0;
            int i_ = 0;

            x = (double[])x.Clone();
            y = (double[])y.Clone();
            w = (double[])w.Clone();
            xc = (double[])xc.Clone();
            yc = (double[])yc.Clone();
            info = 0;

            alglib.ap.assert(st==0 || st==1, "Spline1DFit: internal error!");
            if( st==0 && m<4 )
            {
                info = -1;
                return;
            }
            if( st==1 && m<4 )
            {
                info = -1;
                return;
            }
            if( (n<1 || k<0) || k>=m )
            {
                info = -1;
                return;
            }
            for(i=0; i<=k-1; i++)
            {
                info = 0;
                if( dc[i]<0 )
                {
                    info = -1;
                }
                if( dc[i]>1 )
                {
                    info = -1;
                }
                if( info<0 )
                {
                    return;
                }
            }
            if( st==1 && m%2!=0 )
            {
                
                //
                // Hermite fitter must have even number of basis functions
                //
                info = -2;
                return;
            }
            
            //
            // weight decay for correct handling of task which becomes
            // degenerate after constraints are applied
            //
            decay = 10000*math.machineepsilon;
            
            //
            // Scale X, Y, XC, YC
            //
            lsfitscalexy(ref x, ref y, ref w, n, ref xc, ref yc, dc, k, ref xa, ref xb, ref sa, ref sb, ref xoriginal, ref yoriginal);
            
            //
            // allocate space, initialize:
            // * SX     -   grid for basis functions
            // * SY     -   values of basis functions at grid points
            // * FMatrix-   values of basis functions at X[]
            // * CMatrix-   values (derivatives) of basis functions at XC[]
            //
            y2 = new double[n+m];
            w2 = new double[n+m];
            fmatrix = new double[n+m, m];
            if( k>0 )
            {
                cmatrix = new double[k, m+1];
            }
            if( st==0 )
            {
                
                //
                // allocate space for cubic spline
                //
                sx = new double[m-2];
                sy = new double[m-2];
                for(j=0; j<=m-2-1; j++)
                {
                    sx[j] = (double)(2*j)/(double)(m-2-1)-1;
                }
            }
            if( st==1 )
            {
                
                //
                // allocate space for Hermite spline
                //
                sx = new double[m/2];
                sy = new double[m/2];
                sd = new double[m/2];
                for(j=0; j<=m/2-1; j++)
                {
                    sx[j] = (double)(2*j)/(double)(m/2-1)-1;
                }
            }
            
            //
            // Prepare design and constraints matrices:
            // * fill constraints matrix
            // * fill first N rows of design matrix with values
            // * fill next M rows of design matrix with regularizing term
            // * append M zeros to Y
            // * append M elements, mean(abs(W)) each, to W
            //
            for(j=0; j<=m-1; j++)
            {
                
                //
                // prepare Jth basis function
                //
                if( st==0 )
                {
                    
                    //
                    // cubic spline basis
                    //
                    for(i=0; i<=m-2-1; i++)
                    {
                        sy[i] = 0;
                    }
                    bl = 0;
                    br = 0;
                    if( j<m-2 )
                    {
                        sy[j] = 1;
                    }
                    if( j==m-2 )
                    {
                        bl = 1;
                    }
                    if( j==m-1 )
                    {
                        br = 1;
                    }
                    spline1d.spline1dbuildcubic(sx, sy, m-2, 1, bl, 1, br, s2);
                }
                if( st==1 )
                {
                    
                    //
                    // Hermite basis
                    //
                    for(i=0; i<=m/2-1; i++)
                    {
                        sy[i] = 0;
                        sd[i] = 0;
                    }
                    if( j%2==0 )
                    {
                        sy[j/2] = 1;
                    }
                    else
                    {
                        sd[j/2] = 1;
                    }
                    spline1d.spline1dbuildhermite(sx, sy, sd, m/2, s2);
                }
                
                //
                // values at X[], XC[]
                //
                for(i=0; i<=n-1; i++)
                {
                    fmatrix[i,j] = spline1d.spline1dcalc(s2, x[i]);
                }
                for(i=0; i<=k-1; i++)
                {
                    alglib.ap.assert(dc[i]>=0 && dc[i]<=2, "Spline1DFit: internal error!");
                    spline1d.spline1ddiff(s2, xc[i], ref v0, ref v1, ref v2);
                    if( dc[i]==0 )
                    {
                        cmatrix[i,j] = v0;
                    }
                    if( dc[i]==1 )
                    {
                        cmatrix[i,j] = v1;
                    }
                    if( dc[i]==2 )
                    {
                        cmatrix[i,j] = v2;
                    }
                }
            }
            for(i=0; i<=k-1; i++)
            {
                cmatrix[i,m] = yc[i];
            }
            for(i=0; i<=m-1; i++)
            {
                for(j=0; j<=m-1; j++)
                {
                    if( i==j )
                    {
                        fmatrix[n+i,j] = decay;
                    }
                    else
                    {
                        fmatrix[n+i,j] = 0;
                    }
                }
            }
            y2 = new double[n+m];
            w2 = new double[n+m];
            for(i_=0; i_<=n-1;i_++)
            {
                y2[i_] = y[i_];
            }
            for(i_=0; i_<=n-1;i_++)
            {
                w2[i_] = w[i_];
            }
            mx = 0;
            for(i=0; i<=n-1; i++)
            {
                mx = mx+Math.Abs(w[i]);
            }
            mx = mx/n;
            for(i=0; i<=m-1; i++)
            {
                y2[n+i] = 0;
                w2[n+i] = mx;
            }
            
            //
            // Solve constrained task
            //
            if( k>0 )
            {
                
                //
                // solve using regularization
                //
                lsfitlinearwc(y2, w2, fmatrix, cmatrix, n+m, m, k, ref info, ref tmp, lrep);
            }
            else
            {
                
                //
                // no constraints, no regularization needed
                //
                lsfitlinearwc(y, w, fmatrix, cmatrix, n, m, k, ref info, ref tmp, lrep);
            }
            if( info<0 )
            {
                return;
            }
            
            //
            // Generate spline and scale it
            //
            if( st==0 )
            {
                
                //
                // cubic spline basis
                //
                for(i_=0; i_<=m-2-1;i_++)
                {
                    sy[i_] = tmp[i_];
                }
                spline1d.spline1dbuildcubic(sx, sy, m-2, 1, tmp[m-2], 1, tmp[m-1], s);
            }
            if( st==1 )
            {
                
                //
                // Hermite basis
                //
                for(i=0; i<=m/2-1; i++)
                {
                    sy[i] = tmp[2*i];
                    sd[i] = tmp[2*i+1];
                }
                spline1d.spline1dbuildhermite(sx, sy, sd, m/2, s);
            }
            spline1d.spline1dlintransx(s, 2/(xb-xa), -((xa+xb)/(xb-xa)));
            spline1d.spline1dlintransy(s, sb-sa, sa);
            
            //
            // Scale absolute errors obtained from LSFitLinearW.
            // Relative error should be calculated separately
            // (because of shifting/scaling of the task)
            //
            rep.taskrcond = lrep.taskrcond;
            rep.rmserror = lrep.rmserror*(sb-sa);
            rep.avgerror = lrep.avgerror*(sb-sa);
            rep.maxerror = lrep.maxerror*(sb-sa);
            rep.avgrelerror = 0;
            relcnt = 0;
            for(i=0; i<=n-1; i++)
            {
                if( (double)(yoriginal[i])!=(double)(0) )
                {
                    rep.avgrelerror = rep.avgrelerror+Math.Abs(spline1d.spline1dcalc(s, xoriginal[i])-yoriginal[i])/Math.Abs(yoriginal[i]);
                    relcnt = relcnt+1;
                }
            }
            if( relcnt!=0 )
            {
                rep.avgrelerror = rep.avgrelerror/relcnt;
            }
        }


        /*************************************************************************
        Internal fitting subroutine
        *************************************************************************/
        private static void lsfitlinearinternal(double[] y,
            double[] w,
            double[,] fmatrix,
            int n,
            int m,
            ref int info,
            ref double[] c,
            lsfitreport rep)
        {
            double threshold = 0;
            double[,] ft = new double[0,0];
            double[,] q = new double[0,0];
            double[,] l = new double[0,0];
            double[,] r = new double[0,0];
            double[] b = new double[0];
            double[] wmod = new double[0];
            double[] tau = new double[0];
            int i = 0;
            int j = 0;
            double v = 0;
            double[] sv = new double[0];
            double[,] u = new double[0,0];
            double[,] vt = new double[0,0];
            double[] tmp = new double[0];
            double[] utb = new double[0];
            double[] sutb = new double[0];
            int relcnt = 0;
            int i_ = 0;

            info = 0;
            c = new double[0];

            if( n<1 || m<1 )
            {
                info = -1;
                return;
            }
            info = 1;
            threshold = Math.Sqrt(math.machineepsilon);
            
            //
            // Degenerate case, needs special handling
            //
            if( n<m )
            {
                
                //
                // Create design matrix.
                //
                ft = new double[n, m];
                b = new double[n];
                wmod = new double[n];
                for(j=0; j<=n-1; j++)
                {
                    v = w[j];
                    for(i_=0; i_<=m-1;i_++)
                    {
                        ft[j,i_] = v*fmatrix[j,i_];
                    }
                    b[j] = w[j]*y[j];
                    wmod[j] = 1;
                }
                
                //
                // LQ decomposition and reduction to M=N
                //
                c = new double[m];
                for(i=0; i<=m-1; i++)
                {
                    c[i] = 0;
                }
                rep.taskrcond = 0;
                ortfac.rmatrixlq(ref ft, n, m, ref tau);
                ortfac.rmatrixlqunpackq(ft, n, m, tau, n, ref q);
                ortfac.rmatrixlqunpackl(ft, n, m, ref l);
                lsfitlinearinternal(b, wmod, l, n, n, ref info, ref tmp, rep);
                if( info<=0 )
                {
                    return;
                }
                for(i=0; i<=n-1; i++)
                {
                    v = tmp[i];
                    for(i_=0; i_<=m-1;i_++)
                    {
                        c[i_] = c[i_] + v*q[i,i_];
                    }
                }
                return;
            }
            
            //
            // N>=M. Generate design matrix and reduce to N=M using
            // QR decomposition.
            //
            ft = new double[n, m];
            b = new double[n];
            for(j=0; j<=n-1; j++)
            {
                v = w[j];
                for(i_=0; i_<=m-1;i_++)
                {
                    ft[j,i_] = v*fmatrix[j,i_];
                }
                b[j] = w[j]*y[j];
            }
            ortfac.rmatrixqr(ref ft, n, m, ref tau);
            ortfac.rmatrixqrunpackq(ft, n, m, tau, m, ref q);
            ortfac.rmatrixqrunpackr(ft, n, m, ref r);
            tmp = new double[m];
            for(i=0; i<=m-1; i++)
            {
                tmp[i] = 0;
            }
            for(i=0; i<=n-1; i++)
            {
                v = b[i];
                for(i_=0; i_<=m-1;i_++)
                {
                    tmp[i_] = tmp[i_] + v*q[i,i_];
                }
            }
            b = new double[m];
            for(i_=0; i_<=m-1;i_++)
            {
                b[i_] = tmp[i_];
            }
            
            //
            // R contains reduced MxM design upper triangular matrix,
            // B contains reduced Mx1 right part.
            //
            // Determine system condition number and decide
            // should we use triangular solver (faster) or
            // SVD-based solver (more stable).
            //
            // We can use LU-based RCond estimator for this task.
            //
            rep.taskrcond = rcond.rmatrixlurcondinf(r, m);
            if( (double)(rep.taskrcond)>(double)(threshold) )
            {
                
                //
                // use QR-based solver
                //
                c = new double[m];
                c[m-1] = b[m-1]/r[m-1,m-1];
                for(i=m-2; i>=0; i--)
                {
                    v = 0.0;
                    for(i_=i+1; i_<=m-1;i_++)
                    {
                        v += r[i,i_]*c[i_];
                    }
                    c[i] = (b[i]-v)/r[i,i];
                }
            }
            else
            {
                
                //
                // use SVD-based solver
                //
                if( !svd.rmatrixsvd(r, m, m, 1, 1, 2, ref sv, ref u, ref vt) )
                {
                    info = -4;
                    return;
                }
                utb = new double[m];
                sutb = new double[m];
                for(i=0; i<=m-1; i++)
                {
                    utb[i] = 0;
                }
                for(i=0; i<=m-1; i++)
                {
                    v = b[i];
                    for(i_=0; i_<=m-1;i_++)
                    {
                        utb[i_] = utb[i_] + v*u[i,i_];
                    }
                }
                if( (double)(sv[0])>(double)(0) )
                {
                    rep.taskrcond = sv[m-1]/sv[0];
                    for(i=0; i<=m-1; i++)
                    {
                        if( (double)(sv[i])>(double)(threshold*sv[0]) )
                        {
                            sutb[i] = utb[i]/sv[i];
                        }
                        else
                        {
                            sutb[i] = 0;
                        }
                    }
                }
                else
                {
                    rep.taskrcond = 0;
                    for(i=0; i<=m-1; i++)
                    {
                        sutb[i] = 0;
                    }
                }
                c = new double[m];
                for(i=0; i<=m-1; i++)
                {
                    c[i] = 0;
                }
                for(i=0; i<=m-1; i++)
                {
                    v = sutb[i];
                    for(i_=0; i_<=m-1;i_++)
                    {
                        c[i_] = c[i_] + v*vt[i,i_];
                    }
                }
            }
            
            //
            // calculate errors
            //
            rep.rmserror = 0;
            rep.avgerror = 0;
            rep.avgrelerror = 0;
            rep.maxerror = 0;
            relcnt = 0;
            for(i=0; i<=n-1; i++)
            {
                v = 0.0;
                for(i_=0; i_<=m-1;i_++)
                {
                    v += fmatrix[i,i_]*c[i_];
                }
                rep.rmserror = rep.rmserror+math.sqr(v-y[i]);
                rep.avgerror = rep.avgerror+Math.Abs(v-y[i]);
                if( (double)(y[i])!=(double)(0) )
                {
                    rep.avgrelerror = rep.avgrelerror+Math.Abs(v-y[i])/Math.Abs(y[i]);
                    relcnt = relcnt+1;
                }
                rep.maxerror = Math.Max(rep.maxerror, Math.Abs(v-y[i]));
            }
            rep.rmserror = Math.Sqrt(rep.rmserror/n);
            rep.avgerror = rep.avgerror/n;
            if( relcnt!=0 )
            {
                rep.avgrelerror = rep.avgrelerror/relcnt;
            }
        }


        /*************************************************************************
        Internal subroutine
        *************************************************************************/
        private static void lsfitclearrequestfields(lsfitstate state)
        {
            state.needf = false;
            state.needfg = false;
            state.needfgh = false;
            state.xupdated = false;
        }


        /*************************************************************************
        Internal subroutine, calculates barycentric basis functions.
        Used for efficient simultaneous calculation of N basis functions.

          -- ALGLIB --
             Copyright 17.08.2009 by Bochkanov Sergey
        *************************************************************************/
        private static void barycentriccalcbasis(ratint.barycentricinterpolant b,
            double t,
            ref double[] y)
        {
            double s2 = 0;
            double s = 0;
            double v = 0;
            int i = 0;
            int j = 0;
            int i_ = 0;

            
            //
            // special case: N=1
            //
            if( b.n==1 )
            {
                y[0] = 1;
                return;
            }
            
            //
            // Here we assume that task is normalized, i.e.:
            // 1. abs(Y[i])<=1
            // 2. abs(W[i])<=1
            // 3. X[] is ordered
            //
            // First, we decide: should we use "safe" formula (guarded
            // against overflow) or fast one?
            //
            s = Math.Abs(t-b.x[0]);
            for(i=0; i<=b.n-1; i++)
            {
                v = b.x[i];
                if( (double)(v)==(double)(t) )
                {
                    for(j=0; j<=b.n-1; j++)
                    {
                        y[j] = 0;
                    }
                    y[i] = 1;
                    return;
                }
                v = Math.Abs(t-v);
                if( (double)(v)<(double)(s) )
                {
                    s = v;
                }
            }
            s2 = 0;
            for(i=0; i<=b.n-1; i++)
            {
                v = s/(t-b.x[i]);
                v = v*b.w[i];
                y[i] = v;
                s2 = s2+v;
            }
            v = 1/s2;
            for(i_=0; i_<=b.n-1;i_++)
            {
                y[i_] = v*y[i_];
            }
        }


        /*************************************************************************
        This is internal function for Chebyshev fitting.

        It assumes that input data are normalized:
        * X/XC belong to [-1,+1],
        * mean(Y)=0, stddev(Y)=1.

        It does not checks inputs for errors.

        This function is used to fit general (shifted) Chebyshev models, power
        basis models or barycentric models.

        INPUT PARAMETERS:
            X   -   points, array[0..N-1].
            Y   -   function values, array[0..N-1].
            W   -   weights, array[0..N-1]
            N   -   number of points, N>0.
            XC  -   points where polynomial values/derivatives are constrained,
                    array[0..K-1].
            YC  -   values of constraints, array[0..K-1]
            DC  -   array[0..K-1], types of constraints:
                    * DC[i]=0   means that P(XC[i])=YC[i]
                    * DC[i]=1   means that P'(XC[i])=YC[i]
            K   -   number of constraints, 0<=K<M.
                    K=0 means no constraints (XC/YC/DC are not used in such cases)
            M   -   number of basis functions (= polynomial_degree + 1), M>=1

        OUTPUT PARAMETERS:
            Info-   same format as in LSFitLinearW() subroutine:
                    * Info>0    task is solved
                    * Info<=0   an error occured:
                                -4 means inconvergence of internal SVD
                                -3 means inconsistent constraints
            C   -   interpolant in Chebyshev form; [-1,+1] is used as base interval
            Rep -   report, same format as in LSFitLinearW() subroutine.
                    Following fields are set:
                    * RMSError      rms error on the (X,Y).
                    * AvgError      average error on the (X,Y).
                    * AvgRelError   average relative error on the non-zero Y
                    * MaxError      maximum error
                                    NON-WEIGHTED ERRORS ARE CALCULATED

        IMPORTANT:
            this subroitine doesn't calculate task's condition number for K<>0.

          -- ALGLIB PROJECT --
             Copyright 10.12.2009 by Bochkanov Sergey
        *************************************************************************/
        private static void internalchebyshevfit(double[] x,
            double[] y,
            double[] w,
            int n,
            double[] xc,
            double[] yc,
            int[] dc,
            int k,
            int m,
            ref int info,
            ref double[] c,
            lsfitreport rep)
        {
            double[] y2 = new double[0];
            double[] w2 = new double[0];
            double[] tmp = new double[0];
            double[] tmp2 = new double[0];
            double[] tmpdiff = new double[0];
            double[] bx = new double[0];
            double[] by = new double[0];
            double[] bw = new double[0];
            double[,] fmatrix = new double[0,0];
            double[,] cmatrix = new double[0,0];
            int i = 0;
            int j = 0;
            double mx = 0;
            double decay = 0;
            int i_ = 0;

            xc = (double[])xc.Clone();
            yc = (double[])yc.Clone();
            info = 0;
            c = new double[0];

            
            //
            // weight decay for correct handling of task which becomes
            // degenerate after constraints are applied
            //
            decay = 10000*math.machineepsilon;
            
            //
            // allocate space, initialize/fill:
            // * FMatrix-   values of basis functions at X[]
            // * CMatrix-   values (derivatives) of basis functions at XC[]
            // * fill constraints matrix
            // * fill first N rows of design matrix with values
            // * fill next M rows of design matrix with regularizing term
            // * append M zeros to Y
            // * append M elements, mean(abs(W)) each, to W
            //
            y2 = new double[n+m];
            w2 = new double[n+m];
            tmp = new double[m];
            tmpdiff = new double[m];
            fmatrix = new double[n+m, m];
            if( k>0 )
            {
                cmatrix = new double[k, m+1];
            }
            
            //
            // Fill design matrix, Y2, W2:
            // * first N rows with basis functions for original points
            // * next M rows with decay terms
            //
            for(i=0; i<=n-1; i++)
            {
                
                //
                // prepare Ith row
                // use Tmp for calculations to avoid multidimensional arrays overhead
                //
                for(j=0; j<=m-1; j++)
                {
                    if( j==0 )
                    {
                        tmp[j] = 1;
                    }
                    else
                    {
                        if( j==1 )
                        {
                            tmp[j] = x[i];
                        }
                        else
                        {
                            tmp[j] = 2*x[i]*tmp[j-1]-tmp[j-2];
                        }
                    }
                }
                for(i_=0; i_<=m-1;i_++)
                {
                    fmatrix[i,i_] = tmp[i_];
                }
            }
            for(i=0; i<=m-1; i++)
            {
                for(j=0; j<=m-1; j++)
                {
                    if( i==j )
                    {
                        fmatrix[n+i,j] = decay;
                    }
                    else
                    {
                        fmatrix[n+i,j] = 0;
                    }
                }
            }
            for(i_=0; i_<=n-1;i_++)
            {
                y2[i_] = y[i_];
            }
            for(i_=0; i_<=n-1;i_++)
            {
                w2[i_] = w[i_];
            }
            mx = 0;
            for(i=0; i<=n-1; i++)
            {
                mx = mx+Math.Abs(w[i]);
            }
            mx = mx/n;
            for(i=0; i<=m-1; i++)
            {
                y2[n+i] = 0;
                w2[n+i] = mx;
            }
            
            //
            // fill constraints matrix
            //
            for(i=0; i<=k-1; i++)
            {
                
                //
                // prepare Ith row
                // use Tmp for basis function values,
                // TmpDiff for basos function derivatives
                //
                for(j=0; j<=m-1; j++)
                {
                    if( j==0 )
                    {
                        tmp[j] = 1;
                        tmpdiff[j] = 0;
                    }
                    else
                    {
                        if( j==1 )
                        {
                            tmp[j] = xc[i];
                            tmpdiff[j] = 1;
                        }
                        else
                        {
                            tmp[j] = 2*xc[i]*tmp[j-1]-tmp[j-2];
                            tmpdiff[j] = 2*(tmp[j-1]+xc[i]*tmpdiff[j-1])-tmpdiff[j-2];
                        }
                    }
                }
                if( dc[i]==0 )
                {
                    for(i_=0; i_<=m-1;i_++)
                    {
                        cmatrix[i,i_] = tmp[i_];
                    }
                }
                if( dc[i]==1 )
                {
                    for(i_=0; i_<=m-1;i_++)
                    {
                        cmatrix[i,i_] = tmpdiff[i_];
                    }
                }
                cmatrix[i,m] = yc[i];
            }
            
            //
            // Solve constrained task
            //
            if( k>0 )
            {
                
                //
                // solve using regularization
                //
                lsfitlinearwc(y2, w2, fmatrix, cmatrix, n+m, m, k, ref info, ref c, rep);
            }
            else
            {
                
                //
                // no constraints, no regularization needed
                //
                lsfitlinearwc(y, w, fmatrix, cmatrix, n, m, 0, ref info, ref c, rep);
            }
            if( info<0 )
            {
                return;
            }
        }


        /*************************************************************************
        Internal Floater-Hormann fitting subroutine for fixed D
        *************************************************************************/
        private static void barycentricfitwcfixedd(double[] x,
            double[] y,
            double[] w,
            int n,
            double[] xc,
            double[] yc,
            int[] dc,
            int k,
            int m,
            int d,
            ref int info,
            ratint.barycentricinterpolant b,
            barycentricfitreport rep)
        {
            double[,] fmatrix = new double[0,0];
            double[,] cmatrix = new double[0,0];
            double[] y2 = new double[0];
            double[] w2 = new double[0];
            double[] sx = new double[0];
            double[] sy = new double[0];
            double[] sbf = new double[0];
            double[] xoriginal = new double[0];
            double[] yoriginal = new double[0];
            double[] tmp = new double[0];
            lsfitreport lrep = new lsfitreport();
            double v0 = 0;
            double v1 = 0;
            double mx = 0;
            ratint.barycentricinterpolant b2 = new ratint.barycentricinterpolant();
            int i = 0;
            int j = 0;
            int relcnt = 0;
            double xa = 0;
            double xb = 0;
            double sa = 0;
            double sb = 0;
            double decay = 0;
            int i_ = 0;

            x = (double[])x.Clone();
            y = (double[])y.Clone();
            w = (double[])w.Clone();
            xc = (double[])xc.Clone();
            yc = (double[])yc.Clone();
            info = 0;

            if( ((n<1 || m<2) || k<0) || k>=m )
            {
                info = -1;
                return;
            }
            for(i=0; i<=k-1; i++)
            {
                info = 0;
                if( dc[i]<0 )
                {
                    info = -1;
                }
                if( dc[i]>1 )
                {
                    info = -1;
                }
                if( info<0 )
                {
                    return;
                }
            }
            
            //
            // weight decay for correct handling of task which becomes
            // degenerate after constraints are applied
            //
            decay = 10000*math.machineepsilon;
            
            //
            // Scale X, Y, XC, YC
            //
            lsfitscalexy(ref x, ref y, ref w, n, ref xc, ref yc, dc, k, ref xa, ref xb, ref sa, ref sb, ref xoriginal, ref yoriginal);
            
            //
            // allocate space, initialize:
            // * FMatrix-   values of basis functions at X[]
            // * CMatrix-   values (derivatives) of basis functions at XC[]
            //
            y2 = new double[n+m];
            w2 = new double[n+m];
            fmatrix = new double[n+m, m];
            if( k>0 )
            {
                cmatrix = new double[k, m+1];
            }
            y2 = new double[n+m];
            w2 = new double[n+m];
            
            //
            // Prepare design and constraints matrices:
            // * fill constraints matrix
            // * fill first N rows of design matrix with values
            // * fill next M rows of design matrix with regularizing term
            // * append M zeros to Y
            // * append M elements, mean(abs(W)) each, to W
            //
            sx = new double[m];
            sy = new double[m];
            sbf = new double[m];
            for(j=0; j<=m-1; j++)
            {
                sx[j] = (double)(2*j)/(double)(m-1)-1;
            }
            for(i=0; i<=m-1; i++)
            {
                sy[i] = 1;
            }
            ratint.barycentricbuildfloaterhormann(sx, sy, m, d, b2);
            mx = 0;
            for(i=0; i<=n-1; i++)
            {
                barycentriccalcbasis(b2, x[i], ref sbf);
                for(i_=0; i_<=m-1;i_++)
                {
                    fmatrix[i,i_] = sbf[i_];
                }
                y2[i] = y[i];
                w2[i] = w[i];
                mx = mx+Math.Abs(w[i])/n;
            }
            for(i=0; i<=m-1; i++)
            {
                for(j=0; j<=m-1; j++)
                {
                    if( i==j )
                    {
                        fmatrix[n+i,j] = decay;
                    }
                    else
                    {
                        fmatrix[n+i,j] = 0;
                    }
                }
                y2[n+i] = 0;
                w2[n+i] = mx;
            }
            if( k>0 )
            {
                for(j=0; j<=m-1; j++)
                {
                    for(i=0; i<=m-1; i++)
                    {
                        sy[i] = 0;
                    }
                    sy[j] = 1;
                    ratint.barycentricbuildfloaterhormann(sx, sy, m, d, b2);
                    for(i=0; i<=k-1; i++)
                    {
                        alglib.ap.assert(dc[i]>=0 && dc[i]<=1, "BarycentricFit: internal error!");
                        ratint.barycentricdiff1(b2, xc[i], ref v0, ref v1);
                        if( dc[i]==0 )
                        {
                            cmatrix[i,j] = v0;
                        }
                        if( dc[i]==1 )
                        {
                            cmatrix[i,j] = v1;
                        }
                    }
                }
                for(i=0; i<=k-1; i++)
                {
                    cmatrix[i,m] = yc[i];
                }
            }
            
            //
            // Solve constrained task
            //
            if( k>0 )
            {
                
                //
                // solve using regularization
                //
                lsfitlinearwc(y2, w2, fmatrix, cmatrix, n+m, m, k, ref info, ref tmp, lrep);
            }
            else
            {
                
                //
                // no constraints, no regularization needed
                //
                lsfitlinearwc(y, w, fmatrix, cmatrix, n, m, k, ref info, ref tmp, lrep);
            }
            if( info<0 )
            {
                return;
            }
            
            //
            // Generate interpolant and scale it
            //
            for(i_=0; i_<=m-1;i_++)
            {
                sy[i_] = tmp[i_];
            }
            ratint.barycentricbuildfloaterhormann(sx, sy, m, d, b);
            ratint.barycentriclintransx(b, 2/(xb-xa), -((xa+xb)/(xb-xa)));
            ratint.barycentriclintransy(b, sb-sa, sa);
            
            //
            // Scale absolute errors obtained from LSFitLinearW.
            // Relative error should be calculated separately
            // (because of shifting/scaling of the task)
            //
            rep.taskrcond = lrep.taskrcond;
            rep.rmserror = lrep.rmserror*(sb-sa);
            rep.avgerror = lrep.avgerror*(sb-sa);
            rep.maxerror = lrep.maxerror*(sb-sa);
            rep.avgrelerror = 0;
            relcnt = 0;
            for(i=0; i<=n-1; i++)
            {
                if( (double)(yoriginal[i])!=(double)(0) )
                {
                    rep.avgrelerror = rep.avgrelerror+Math.Abs(ratint.barycentriccalc(b, xoriginal[i])-yoriginal[i])/Math.Abs(yoriginal[i]);
                    relcnt = relcnt+1;
                }
            }
            if( relcnt!=0 )
            {
                rep.avgrelerror = rep.avgrelerror/relcnt;
            }
        }


    }
    public class pspline
    {
        /*************************************************************************
        Parametric spline inteprolant: 2-dimensional curve.

        You should not try to access its members directly - use PSpline2XXXXXXXX()
        functions instead.
        *************************************************************************/
        public class pspline2interpolant
        {
            public int n;
            public bool periodic;
            public double[] p;
            public spline1d.spline1dinterpolant x;
            public spline1d.spline1dinterpolant y;
            public pspline2interpolant()
            {
                p = new double[0];
                x = new spline1d.spline1dinterpolant();
                y = new spline1d.spline1dinterpolant();
            }
        };


        /*************************************************************************
        Parametric spline inteprolant: 3-dimensional curve.

        You should not try to access its members directly - use PSpline3XXXXXXXX()
        functions instead.
        *************************************************************************/
        public class pspline3interpolant
        {
            public int n;
            public bool periodic;
            public double[] p;
            public spline1d.spline1dinterpolant x;
            public spline1d.spline1dinterpolant y;
            public spline1d.spline1dinterpolant z;
            public pspline3interpolant()
            {
                p = new double[0];
                x = new spline1d.spline1dinterpolant();
                y = new spline1d.spline1dinterpolant();
                z = new spline1d.spline1dinterpolant();
            }
        };




        /*************************************************************************
        This function  builds  non-periodic 2-dimensional parametric spline  which
        starts at (X[0],Y[0]) and ends at (X[N-1],Y[N-1]).

        INPUT PARAMETERS:
            XY  -   points, array[0..N-1,0..1].
                    XY[I,0:1] corresponds to the Ith point.
                    Order of points is important!
            N   -   points count, N>=5 for Akima splines, N>=2 for other types  of
                    splines.
            ST  -   spline type:
                    * 0     Akima spline
                    * 1     parabolically terminated Catmull-Rom spline (Tension=0)
                    * 2     parabolically terminated cubic spline
            PT  -   parameterization type:
                    * 0     uniform
                    * 1     chord length
                    * 2     centripetal

        OUTPUT PARAMETERS:
            P   -   parametric spline interpolant


        NOTES:
        * this function  assumes  that  there all consequent points  are distinct.
          I.e. (x0,y0)<>(x1,y1),  (x1,y1)<>(x2,y2),  (x2,y2)<>(x3,y3)  and  so on.
          However, non-consequent points may coincide, i.e. we can  have  (x0,y0)=
          =(x2,y2).

          -- ALGLIB PROJECT --
             Copyright 28.05.2010 by Bochkanov Sergey
        *************************************************************************/
        public static void pspline2build(double[,] xy,
            int n,
            int st,
            int pt,
            pspline2interpolant p)
        {
            double[] tmp = new double[0];
            int i_ = 0;

            xy = (double[,])xy.Clone();

            alglib.ap.assert(st>=0 && st<=2, "PSpline2Build: incorrect spline type!");
            alglib.ap.assert(pt>=0 && pt<=2, "PSpline2Build: incorrect parameterization type!");
            if( st==0 )
            {
                alglib.ap.assert(n>=5, "PSpline2Build: N<5 (minimum value for Akima splines)!");
            }
            else
            {
                alglib.ap.assert(n>=2, "PSpline2Build: N<2!");
            }
            
            //
            // Prepare
            //
            p.n = n;
            p.periodic = false;
            tmp = new double[n];
            
            //
            // Build parameterization, check that all parameters are distinct
            //
            pspline2par(xy, n, pt, ref p.p);
            alglib.ap.assert(apserv.aredistinct(p.p, n), "PSpline2Build: consequent points are too close!");
            
            //
            // Build splines
            //
            if( st==0 )
            {
                for(i_=0; i_<=n-1;i_++)
                {
                    tmp[i_] = xy[i_,0];
                }
                spline1d.spline1dbuildakima(p.p, tmp, n, p.x);
                for(i_=0; i_<=n-1;i_++)
                {
                    tmp[i_] = xy[i_,1];
                }
                spline1d.spline1dbuildakima(p.p, tmp, n, p.y);
            }
            if( st==1 )
            {
                for(i_=0; i_<=n-1;i_++)
                {
                    tmp[i_] = xy[i_,0];
                }
                spline1d.spline1dbuildcatmullrom(p.p, tmp, n, 0, 0.0, p.x);
                for(i_=0; i_<=n-1;i_++)
                {
                    tmp[i_] = xy[i_,1];
                }
                spline1d.spline1dbuildcatmullrom(p.p, tmp, n, 0, 0.0, p.y);
            }
            if( st==2 )
            {
                for(i_=0; i_<=n-1;i_++)
                {
                    tmp[i_] = xy[i_,0];
                }
                spline1d.spline1dbuildcubic(p.p, tmp, n, 0, 0.0, 0, 0.0, p.x);
                for(i_=0; i_<=n-1;i_++)
                {
                    tmp[i_] = xy[i_,1];
                }
                spline1d.spline1dbuildcubic(p.p, tmp, n, 0, 0.0, 0, 0.0, p.y);
            }
        }


        /*************************************************************************
        This function  builds  non-periodic 3-dimensional parametric spline  which
        starts at (X[0],Y[0],Z[0]) and ends at (X[N-1],Y[N-1],Z[N-1]).

        Same as PSpline2Build() function, but for 3D, so we  won't  duplicate  its
        description here.

          -- ALGLIB PROJECT --
             Copyright 28.05.2010 by Bochkanov Sergey
        *************************************************************************/
        public static void pspline3build(double[,] xy,
            int n,
            int st,
            int pt,
            pspline3interpolant p)
        {
            double[] tmp = new double[0];
            int i_ = 0;

            xy = (double[,])xy.Clone();

            alglib.ap.assert(st>=0 && st<=2, "PSpline3Build: incorrect spline type!");
            alglib.ap.assert(pt>=0 && pt<=2, "PSpline3Build: incorrect parameterization type!");
            if( st==0 )
            {
                alglib.ap.assert(n>=5, "PSpline3Build: N<5 (minimum value for Akima splines)!");
            }
            else
            {
                alglib.ap.assert(n>=2, "PSpline3Build: N<2!");
            }
            
            //
            // Prepare
            //
            p.n = n;
            p.periodic = false;
            tmp = new double[n];
            
            //
            // Build parameterization, check that all parameters are distinct
            //
            pspline3par(xy, n, pt, ref p.p);
            alglib.ap.assert(apserv.aredistinct(p.p, n), "PSpline3Build: consequent points are too close!");
            
            //
            // Build splines
            //
            if( st==0 )
            {
                for(i_=0; i_<=n-1;i_++)
                {
                    tmp[i_] = xy[i_,0];
                }
                spline1d.spline1dbuildakima(p.p, tmp, n, p.x);
                for(i_=0; i_<=n-1;i_++)
                {
                    tmp[i_] = xy[i_,1];
                }
                spline1d.spline1dbuildakima(p.p, tmp, n, p.y);
                for(i_=0; i_<=n-1;i_++)
                {
                    tmp[i_] = xy[i_,2];
                }
                spline1d.spline1dbuildakima(p.p, tmp, n, p.z);
            }
            if( st==1 )
            {
                for(i_=0; i_<=n-1;i_++)
                {
                    tmp[i_] = xy[i_,0];
                }
                spline1d.spline1dbuildcatmullrom(p.p, tmp, n, 0, 0.0, p.x);
                for(i_=0; i_<=n-1;i_++)
                {
                    tmp[i_] = xy[i_,1];
                }
                spline1d.spline1dbuildcatmullrom(p.p, tmp, n, 0, 0.0, p.y);
                for(i_=0; i_<=n-1;i_++)
                {
                    tmp[i_] = xy[i_,2];
                }
                spline1d.spline1dbuildcatmullrom(p.p, tmp, n, 0, 0.0, p.z);
            }
            if( st==2 )
            {
                for(i_=0; i_<=n-1;i_++)
                {
                    tmp[i_] = xy[i_,0];
                }
                spline1d.spline1dbuildcubic(p.p, tmp, n, 0, 0.0, 0, 0.0, p.x);
                for(i_=0; i_<=n-1;i_++)
                {
                    tmp[i_] = xy[i_,1];
                }
                spline1d.spline1dbuildcubic(p.p, tmp, n, 0, 0.0, 0, 0.0, p.y);
                for(i_=0; i_<=n-1;i_++)
                {
                    tmp[i_] = xy[i_,2];
                }
                spline1d.spline1dbuildcubic(p.p, tmp, n, 0, 0.0, 0, 0.0, p.z);
            }
        }


        /*************************************************************************
        This  function  builds  periodic  2-dimensional  parametric  spline  which
        starts at (X[0],Y[0]), goes through all points to (X[N-1],Y[N-1]) and then
        back to (X[0],Y[0]).

        INPUT PARAMETERS:
            XY  -   points, array[0..N-1,0..1].
                    XY[I,0:1] corresponds to the Ith point.
                    XY[N-1,0:1] must be different from XY[0,0:1].
                    Order of points is important!
            N   -   points count, N>=3 for other types of splines.
            ST  -   spline type:
                    * 1     Catmull-Rom spline (Tension=0) with cyclic boundary conditions
                    * 2     cubic spline with cyclic boundary conditions
            PT  -   parameterization type:
                    * 0     uniform
                    * 1     chord length
                    * 2     centripetal

        OUTPUT PARAMETERS:
            P   -   parametric spline interpolant


        NOTES:
        * this function  assumes  that there all consequent points  are  distinct.
          I.e. (x0,y0)<>(x1,y1), (x1,y1)<>(x2,y2),  (x2,y2)<>(x3,y3)  and  so  on.
          However, non-consequent points may coincide, i.e. we can  have  (x0,y0)=
          =(x2,y2).
        * last point of sequence is NOT equal to the first  point.  You  shouldn't
          make curve "explicitly periodic" by making them equal.

          -- ALGLIB PROJECT --
             Copyright 28.05.2010 by Bochkanov Sergey
        *************************************************************************/
        public static void pspline2buildperiodic(double[,] xy,
            int n,
            int st,
            int pt,
            pspline2interpolant p)
        {
            double[,] xyp = new double[0,0];
            double[] tmp = new double[0];
            int i_ = 0;

            xy = (double[,])xy.Clone();

            alglib.ap.assert(st>=1 && st<=2, "PSpline2BuildPeriodic: incorrect spline type!");
            alglib.ap.assert(pt>=0 && pt<=2, "PSpline2BuildPeriodic: incorrect parameterization type!");
            alglib.ap.assert(n>=3, "PSpline2BuildPeriodic: N<3!");
            
            //
            // Prepare
            //
            p.n = n;
            p.periodic = true;
            tmp = new double[n+1];
            xyp = new double[n+1, 2];
            for(i_=0; i_<=n-1;i_++)
            {
                xyp[i_,0] = xy[i_,0];
            }
            for(i_=0; i_<=n-1;i_++)
            {
                xyp[i_,1] = xy[i_,1];
            }
            for(i_=0; i_<=1;i_++)
            {
                xyp[n,i_] = xy[0,i_];
            }
            
            //
            // Build parameterization, check that all parameters are distinct
            //
            pspline2par(xyp, n+1, pt, ref p.p);
            alglib.ap.assert(apserv.aredistinct(p.p, n+1), "PSpline2BuildPeriodic: consequent (or first and last) points are too close!");
            
            //
            // Build splines
            //
            if( st==1 )
            {
                for(i_=0; i_<=n;i_++)
                {
                    tmp[i_] = xyp[i_,0];
                }
                spline1d.spline1dbuildcatmullrom(p.p, tmp, n+1, -1, 0.0, p.x);
                for(i_=0; i_<=n;i_++)
                {
                    tmp[i_] = xyp[i_,1];
                }
                spline1d.spline1dbuildcatmullrom(p.p, tmp, n+1, -1, 0.0, p.y);
            }
            if( st==2 )
            {
                for(i_=0; i_<=n;i_++)
                {
                    tmp[i_] = xyp[i_,0];
                }
                spline1d.spline1dbuildcubic(p.p, tmp, n+1, -1, 0.0, -1, 0.0, p.x);
                for(i_=0; i_<=n;i_++)
                {
                    tmp[i_] = xyp[i_,1];
                }
                spline1d.spline1dbuildcubic(p.p, tmp, n+1, -1, 0.0, -1, 0.0, p.y);
            }
        }


        /*************************************************************************
        This  function  builds  periodic  3-dimensional  parametric  spline  which
        starts at (X[0],Y[0],Z[0]), goes through all points to (X[N-1],Y[N-1],Z[N-1])
        and then back to (X[0],Y[0],Z[0]).

        Same as PSpline2Build() function, but for 3D, so we  won't  duplicate  its
        description here.

          -- ALGLIB PROJECT --
             Copyright 28.05.2010 by Bochkanov Sergey
        *************************************************************************/
        public static void pspline3buildperiodic(double[,] xy,
            int n,
            int st,
            int pt,
            pspline3interpolant p)
        {
            double[,] xyp = new double[0,0];
            double[] tmp = new double[0];
            int i_ = 0;

            xy = (double[,])xy.Clone();

            alglib.ap.assert(st>=1 && st<=2, "PSpline3BuildPeriodic: incorrect spline type!");
            alglib.ap.assert(pt>=0 && pt<=2, "PSpline3BuildPeriodic: incorrect parameterization type!");
            alglib.ap.assert(n>=3, "PSpline3BuildPeriodic: N<3!");
            
            //
            // Prepare
            //
            p.n = n;
            p.periodic = true;
            tmp = new double[n+1];
            xyp = new double[n+1, 3];
            for(i_=0; i_<=n-1;i_++)
            {
                xyp[i_,0] = xy[i_,0];
            }
            for(i_=0; i_<=n-1;i_++)
            {
                xyp[i_,1] = xy[i_,1];
            }
            for(i_=0; i_<=n-1;i_++)
            {
                xyp[i_,2] = xy[i_,2];
            }
            for(i_=0; i_<=2;i_++)
            {
                xyp[n,i_] = xy[0,i_];
            }
            
            //
            // Build parameterization, check that all parameters are distinct
            //
            pspline3par(xyp, n+1, pt, ref p.p);
            alglib.ap.assert(apserv.aredistinct(p.p, n+1), "PSplineBuild2Periodic: consequent (or first and last) points are too close!");
            
            //
            // Build splines
            //
            if( st==1 )
            {
                for(i_=0; i_<=n;i_++)
                {
                    tmp[i_] = xyp[i_,0];
                }
                spline1d.spline1dbuildcatmullrom(p.p, tmp, n+1, -1, 0.0, p.x);
                for(i_=0; i_<=n;i_++)
                {
                    tmp[i_] = xyp[i_,1];
                }
                spline1d.spline1dbuildcatmullrom(p.p, tmp, n+1, -1, 0.0, p.y);
                for(i_=0; i_<=n;i_++)
                {
                    tmp[i_] = xyp[i_,2];
                }
                spline1d.spline1dbuildcatmullrom(p.p, tmp, n+1, -1, 0.0, p.z);
            }
            if( st==2 )
            {
                for(i_=0; i_<=n;i_++)
                {
                    tmp[i_] = xyp[i_,0];
                }
                spline1d.spline1dbuildcubic(p.p, tmp, n+1, -1, 0.0, -1, 0.0, p.x);
                for(i_=0; i_<=n;i_++)
                {
                    tmp[i_] = xyp[i_,1];
                }
                spline1d.spline1dbuildcubic(p.p, tmp, n+1, -1, 0.0, -1, 0.0, p.y);
                for(i_=0; i_<=n;i_++)
                {
                    tmp[i_] = xyp[i_,2];
                }
                spline1d.spline1dbuildcubic(p.p, tmp, n+1, -1, 0.0, -1, 0.0, p.z);
            }
        }


        /*************************************************************************
        This function returns vector of parameter values correspoding to points.

        I.e. for P created from (X[0],Y[0])...(X[N-1],Y[N-1]) and U=TValues(P)  we
        have
            (X[0],Y[0]) = PSpline2Calc(P,U[0]),
            (X[1],Y[1]) = PSpline2Calc(P,U[1]),
            (X[2],Y[2]) = PSpline2Calc(P,U[2]),
            ...

        INPUT PARAMETERS:
            P   -   parametric spline interpolant

        OUTPUT PARAMETERS:
            N   -   array size
            T   -   array[0..N-1]


        NOTES:
        * for non-periodic splines U[0]=0, U[0]<U[1]<...<U[N-1], U[N-1]=1
        * for periodic splines     U[0]=0, U[0]<U[1]<...<U[N-1], U[N-1]<1

          -- ALGLIB PROJECT --
             Copyright 28.05.2010 by Bochkanov Sergey
        *************************************************************************/
        public static void pspline2parametervalues(pspline2interpolant p,
            ref int n,
            ref double[] t)
        {
            int i_ = 0;

            n = 0;
            t = new double[0];

            alglib.ap.assert(p.n>=2, "PSpline2ParameterValues: internal error!");
            n = p.n;
            t = new double[n];
            for(i_=0; i_<=n-1;i_++)
            {
                t[i_] = p.p[i_];
            }
            t[0] = 0;
            if( !p.periodic )
            {
                t[n-1] = 1;
            }
        }


        /*************************************************************************
        This function returns vector of parameter values correspoding to points.

        Same as PSpline2ParameterValues(), but for 3D.

          -- ALGLIB PROJECT --
             Copyright 28.05.2010 by Bochkanov Sergey
        *************************************************************************/
        public static void pspline3parametervalues(pspline3interpolant p,
            ref int n,
            ref double[] t)
        {
            int i_ = 0;

            n = 0;
            t = new double[0];

            alglib.ap.assert(p.n>=2, "PSpline3ParameterValues: internal error!");
            n = p.n;
            t = new double[n];
            for(i_=0; i_<=n-1;i_++)
            {
                t[i_] = p.p[i_];
            }
            t[0] = 0;
            if( !p.periodic )
            {
                t[n-1] = 1;
            }
        }


        /*************************************************************************
        This function  calculates  the value of the parametric spline for a  given
        value of parameter T

        INPUT PARAMETERS:
            P   -   parametric spline interpolant
            T   -   point:
                    * T in [0,1] corresponds to interval spanned by points
                    * for non-periodic splines T<0 (or T>1) correspond to parts of
                      the curve before the first (after the last) point
                    * for periodic splines T<0 (or T>1) are projected  into  [0,1]
                      by making T=T-floor(T).

        OUTPUT PARAMETERS:
            X   -   X-position
            Y   -   Y-position


          -- ALGLIB PROJECT --
             Copyright 28.05.2010 by Bochkanov Sergey
        *************************************************************************/
        public static void pspline2calc(pspline2interpolant p,
            double t,
            ref double x,
            ref double y)
        {
            x = 0;
            y = 0;

            if( p.periodic )
            {
                t = t-(int)Math.Floor(t);
            }
            x = spline1d.spline1dcalc(p.x, t);
            y = spline1d.spline1dcalc(p.y, t);
        }


        /*************************************************************************
        This function  calculates  the value of the parametric spline for a  given
        value of parameter T.

        INPUT PARAMETERS:
            P   -   parametric spline interpolant
            T   -   point:
                    * T in [0,1] corresponds to interval spanned by points
                    * for non-periodic splines T<0 (or T>1) correspond to parts of
                      the curve before the first (after the last) point
                    * for periodic splines T<0 (or T>1) are projected  into  [0,1]
                      by making T=T-floor(T).

        OUTPUT PARAMETERS:
            X   -   X-position
            Y   -   Y-position
            Z   -   Z-position


          -- ALGLIB PROJECT --
             Copyright 28.05.2010 by Bochkanov Sergey
        *************************************************************************/
        public static void pspline3calc(pspline3interpolant p,
            double t,
            ref double x,
            ref double y,
            ref double z)
        {
            x = 0;
            y = 0;
            z = 0;

            if( p.periodic )
            {
                t = t-(int)Math.Floor(t);
            }
            x = spline1d.spline1dcalc(p.x, t);
            y = spline1d.spline1dcalc(p.y, t);
            z = spline1d.spline1dcalc(p.z, t);
        }


        /*************************************************************************
        This function  calculates  tangent vector for a given value of parameter T

        INPUT PARAMETERS:
            P   -   parametric spline interpolant
            T   -   point:
                    * T in [0,1] corresponds to interval spanned by points
                    * for non-periodic splines T<0 (or T>1) correspond to parts of
                      the curve before the first (after the last) point
                    * for periodic splines T<0 (or T>1) are projected  into  [0,1]
                      by making T=T-floor(T).

        OUTPUT PARAMETERS:
            X    -   X-component of tangent vector (normalized)
            Y    -   Y-component of tangent vector (normalized)
            
        NOTE:
            X^2+Y^2 is either 1 (for non-zero tangent vector) or 0.


          -- ALGLIB PROJECT --
             Copyright 28.05.2010 by Bochkanov Sergey
        *************************************************************************/
        public static void pspline2tangent(pspline2interpolant p,
            double t,
            ref double x,
            ref double y)
        {
            double v = 0;
            double v0 = 0;
            double v1 = 0;

            x = 0;
            y = 0;

            if( p.periodic )
            {
                t = t-(int)Math.Floor(t);
            }
            pspline2diff(p, t, ref v0, ref x, ref v1, ref y);
            if( (double)(x)!=(double)(0) || (double)(y)!=(double)(0) )
            {
                
                //
                // this code is a bit more complex than X^2+Y^2 to avoid
                // overflow for large values of X and Y.
                //
                v = apserv.safepythag2(x, y);
                x = x/v;
                y = y/v;
            }
        }


        /*************************************************************************
        This function  calculates  tangent vector for a given value of parameter T

        INPUT PARAMETERS:
            P   -   parametric spline interpolant
            T   -   point:
                    * T in [0,1] corresponds to interval spanned by points
                    * for non-periodic splines T<0 (or T>1) correspond to parts of
                      the curve before the first (after the last) point
                    * for periodic splines T<0 (or T>1) are projected  into  [0,1]
                      by making T=T-floor(T).

        OUTPUT PARAMETERS:
            X    -   X-component of tangent vector (normalized)
            Y    -   Y-component of tangent vector (normalized)
            Z    -   Z-component of tangent vector (normalized)

        NOTE:
            X^2+Y^2+Z^2 is either 1 (for non-zero tangent vector) or 0.


          -- ALGLIB PROJECT --
             Copyright 28.05.2010 by Bochkanov Sergey
        *************************************************************************/
        public static void pspline3tangent(pspline3interpolant p,
            double t,
            ref double x,
            ref double y,
            ref double z)
        {
            double v = 0;
            double v0 = 0;
            double v1 = 0;
            double v2 = 0;

            x = 0;
            y = 0;
            z = 0;

            if( p.periodic )
            {
                t = t-(int)Math.Floor(t);
            }
            pspline3diff(p, t, ref v0, ref x, ref v1, ref y, ref v2, ref z);
            if( ((double)(x)!=(double)(0) || (double)(y)!=(double)(0)) || (double)(z)!=(double)(0) )
            {
                v = apserv.safepythag3(x, y, z);
                x = x/v;
                y = y/v;
                z = z/v;
            }
        }


        /*************************************************************************
        This function calculates derivative, i.e. it returns (dX/dT,dY/dT).

        INPUT PARAMETERS:
            P   -   parametric spline interpolant
            T   -   point:
                    * T in [0,1] corresponds to interval spanned by points
                    * for non-periodic splines T<0 (or T>1) correspond to parts of
                      the curve before the first (after the last) point
                    * for periodic splines T<0 (or T>1) are projected  into  [0,1]
                      by making T=T-floor(T).

        OUTPUT PARAMETERS:
            X   -   X-value
            DX  -   X-derivative
            Y   -   Y-value
            DY  -   Y-derivative


          -- ALGLIB PROJECT --
             Copyright 28.05.2010 by Bochkanov Sergey
        *************************************************************************/
        public static void pspline2diff(pspline2interpolant p,
            double t,
            ref double x,
            ref double dx,
            ref double y,
            ref double dy)
        {
            double d2s = 0;

            x = 0;
            dx = 0;
            y = 0;
            dy = 0;

            if( p.periodic )
            {
                t = t-(int)Math.Floor(t);
            }
            spline1d.spline1ddiff(p.x, t, ref x, ref dx, ref d2s);
            spline1d.spline1ddiff(p.y, t, ref y, ref dy, ref d2s);
        }


        /*************************************************************************
        This function calculates derivative, i.e. it returns (dX/dT,dY/dT,dZ/dT).

        INPUT PARAMETERS:
            P   -   parametric spline interpolant
            T   -   point:
                    * T in [0,1] corresponds to interval spanned by points
                    * for non-periodic splines T<0 (or T>1) correspond to parts of
                      the curve before the first (after the last) point
                    * for periodic splines T<0 (or T>1) are projected  into  [0,1]
                      by making T=T-floor(T).

        OUTPUT PARAMETERS:
            X   -   X-value
            DX  -   X-derivative
            Y   -   Y-value
            DY  -   Y-derivative
            Z   -   Z-value
            DZ  -   Z-derivative


          -- ALGLIB PROJECT --
             Copyright 28.05.2010 by Bochkanov Sergey
        *************************************************************************/
        public static void pspline3diff(pspline3interpolant p,
            double t,
            ref double x,
            ref double dx,
            ref double y,
            ref double dy,
            ref double z,
            ref double dz)
        {
            double d2s = 0;

            x = 0;
            dx = 0;
            y = 0;
            dy = 0;
            z = 0;
            dz = 0;

            if( p.periodic )
            {
                t = t-(int)Math.Floor(t);
            }
            spline1d.spline1ddiff(p.x, t, ref x, ref dx, ref d2s);
            spline1d.spline1ddiff(p.y, t, ref y, ref dy, ref d2s);
            spline1d.spline1ddiff(p.z, t, ref z, ref dz, ref d2s);
        }


        /*************************************************************************
        This function calculates first and second derivative with respect to T.

        INPUT PARAMETERS:
            P   -   parametric spline interpolant
            T   -   point:
                    * T in [0,1] corresponds to interval spanned by points
                    * for non-periodic splines T<0 (or T>1) correspond to parts of
                      the curve before the first (after the last) point
                    * for periodic splines T<0 (or T>1) are projected  into  [0,1]
                      by making T=T-floor(T).

        OUTPUT PARAMETERS:
            X   -   X-value
            DX  -   derivative
            D2X -   second derivative
            Y   -   Y-value
            DY  -   derivative
            D2Y -   second derivative


          -- ALGLIB PROJECT --
             Copyright 28.05.2010 by Bochkanov Sergey
        *************************************************************************/
        public static void pspline2diff2(pspline2interpolant p,
            double t,
            ref double x,
            ref double dx,
            ref double d2x,
            ref double y,
            ref double dy,
            ref double d2y)
        {
            x = 0;
            dx = 0;
            d2x = 0;
            y = 0;
            dy = 0;
            d2y = 0;

            if( p.periodic )
            {
                t = t-(int)Math.Floor(t);
            }
            spline1d.spline1ddiff(p.x, t, ref x, ref dx, ref d2x);
            spline1d.spline1ddiff(p.y, t, ref y, ref dy, ref d2y);
        }


        /*************************************************************************
        This function calculates first and second derivative with respect to T.

        INPUT PARAMETERS:
            P   -   parametric spline interpolant
            T   -   point:
                    * T in [0,1] corresponds to interval spanned by points
                    * for non-periodic splines T<0 (or T>1) correspond to parts of
                      the curve before the first (after the last) point
                    * for periodic splines T<0 (or T>1) are projected  into  [0,1]
                      by making T=T-floor(T).

        OUTPUT PARAMETERS:
            X   -   X-value
            DX  -   derivative
            D2X -   second derivative
            Y   -   Y-value
            DY  -   derivative
            D2Y -   second derivative
            Z   -   Z-value
            DZ  -   derivative
            D2Z -   second derivative


          -- ALGLIB PROJECT --
             Copyright 28.05.2010 by Bochkanov Sergey
        *************************************************************************/
        public static void pspline3diff2(pspline3interpolant p,
            double t,
            ref double x,
            ref double dx,
            ref double d2x,
            ref double y,
            ref double dy,
            ref double d2y,
            ref double z,
            ref double dz,
            ref double d2z)
        {
            x = 0;
            dx = 0;
            d2x = 0;
            y = 0;
            dy = 0;
            d2y = 0;
            z = 0;
            dz = 0;
            d2z = 0;

            if( p.periodic )
            {
                t = t-(int)Math.Floor(t);
            }
            spline1d.spline1ddiff(p.x, t, ref x, ref dx, ref d2x);
            spline1d.spline1ddiff(p.y, t, ref y, ref dy, ref d2y);
            spline1d.spline1ddiff(p.z, t, ref z, ref dz, ref d2z);
        }


        /*************************************************************************
        This function  calculates  arc length, i.e. length of  curve  between  t=a
        and t=b.

        INPUT PARAMETERS:
            P   -   parametric spline interpolant
            A,B -   parameter values corresponding to arc ends:
                    * B>A will result in positive length returned
                    * B<A will result in negative length returned

        RESULT:
            length of arc starting at T=A and ending at T=B.


          -- ALGLIB PROJECT --
             Copyright 30.05.2010 by Bochkanov Sergey
        *************************************************************************/
        public static double pspline2arclength(pspline2interpolant p,
            double a,
            double b)
        {
            double result = 0;
            autogk.autogkstate state = new autogk.autogkstate();
            autogk.autogkreport rep = new autogk.autogkreport();
            double sx = 0;
            double dsx = 0;
            double d2sx = 0;
            double sy = 0;
            double dsy = 0;
            double d2sy = 0;

            autogk.autogksmooth(a, b, state);
            while( autogk.autogkiteration(state) )
            {
                spline1d.spline1ddiff(p.x, state.x, ref sx, ref dsx, ref d2sx);
                spline1d.spline1ddiff(p.y, state.x, ref sy, ref dsy, ref d2sy);
                state.f = apserv.safepythag2(dsx, dsy);
            }
            autogk.autogkresults(state, ref result, rep);
            alglib.ap.assert(rep.terminationtype>0, "PSpline2ArcLength: internal error!");
            return result;
        }


        /*************************************************************************
        This function  calculates  arc length, i.e. length of  curve  between  t=a
        and t=b.

        INPUT PARAMETERS:
            P   -   parametric spline interpolant
            A,B -   parameter values corresponding to arc ends:
                    * B>A will result in positive length returned
                    * B<A will result in negative length returned

        RESULT:
            length of arc starting at T=A and ending at T=B.


          -- ALGLIB PROJECT --
             Copyright 30.05.2010 by Bochkanov Sergey
        *************************************************************************/
        public static double pspline3arclength(pspline3interpolant p,
            double a,
            double b)
        {
            double result = 0;
            autogk.autogkstate state = new autogk.autogkstate();
            autogk.autogkreport rep = new autogk.autogkreport();
            double sx = 0;
            double dsx = 0;
            double d2sx = 0;
            double sy = 0;
            double dsy = 0;
            double d2sy = 0;
            double sz = 0;
            double dsz = 0;
            double d2sz = 0;

            autogk.autogksmooth(a, b, state);
            while( autogk.autogkiteration(state) )
            {
                spline1d.spline1ddiff(p.x, state.x, ref sx, ref dsx, ref d2sx);
                spline1d.spline1ddiff(p.y, state.x, ref sy, ref dsy, ref d2sy);
                spline1d.spline1ddiff(p.z, state.x, ref sz, ref dsz, ref d2sz);
                state.f = apserv.safepythag3(dsx, dsy, dsz);
            }
            autogk.autogkresults(state, ref result, rep);
            alglib.ap.assert(rep.terminationtype>0, "PSpline3ArcLength: internal error!");
            return result;
        }


        /*************************************************************************
        Builds non-periodic parameterization for 2-dimensional spline
        *************************************************************************/
        private static void pspline2par(double[,] xy,
            int n,
            int pt,
            ref double[] p)
        {
            double v = 0;
            int i = 0;
            int i_ = 0;

            p = new double[0];

            alglib.ap.assert(pt>=0 && pt<=2, "PSpline2Par: internal error!");
            
            //
            // Build parameterization:
            // * fill by non-normalized values
            // * normalize them so we have P[0]=0, P[N-1]=1.
            //
            p = new double[n];
            if( pt==0 )
            {
                for(i=0; i<=n-1; i++)
                {
                    p[i] = i;
                }
            }
            if( pt==1 )
            {
                p[0] = 0;
                for(i=1; i<=n-1; i++)
                {
                    p[i] = p[i-1]+apserv.safepythag2(xy[i,0]-xy[i-1,0], xy[i,1]-xy[i-1,1]);
                }
            }
            if( pt==2 )
            {
                p[0] = 0;
                for(i=1; i<=n-1; i++)
                {
                    p[i] = p[i-1]+Math.Sqrt(apserv.safepythag2(xy[i,0]-xy[i-1,0], xy[i,1]-xy[i-1,1]));
                }
            }
            v = 1/p[n-1];
            for(i_=0; i_<=n-1;i_++)
            {
                p[i_] = v*p[i_];
            }
        }


        /*************************************************************************
        Builds non-periodic parameterization for 3-dimensional spline
        *************************************************************************/
        private static void pspline3par(double[,] xy,
            int n,
            int pt,
            ref double[] p)
        {
            double v = 0;
            int i = 0;
            int i_ = 0;

            p = new double[0];

            alglib.ap.assert(pt>=0 && pt<=2, "PSpline3Par: internal error!");
            
            //
            // Build parameterization:
            // * fill by non-normalized values
            // * normalize them so we have P[0]=0, P[N-1]=1.
            //
            p = new double[n];
            if( pt==0 )
            {
                for(i=0; i<=n-1; i++)
                {
                    p[i] = i;
                }
            }
            if( pt==1 )
            {
                p[0] = 0;
                for(i=1; i<=n-1; i++)
                {
                    p[i] = p[i-1]+apserv.safepythag3(xy[i,0]-xy[i-1,0], xy[i,1]-xy[i-1,1], xy[i,2]-xy[i-1,2]);
                }
            }
            if( pt==2 )
            {
                p[0] = 0;
                for(i=1; i<=n-1; i++)
                {
                    p[i] = p[i-1]+Math.Sqrt(apserv.safepythag3(xy[i,0]-xy[i-1,0], xy[i,1]-xy[i-1,1], xy[i,2]-xy[i-1,2]));
                }
            }
            v = 1/p[n-1];
            for(i_=0; i_<=n-1;i_++)
            {
                p[i_] = v*p[i_];
            }
        }


    }
    public class rbf
    {
        /*************************************************************************
        RBF model.

        Never try to directly work with fields of this object - always use  ALGLIB
        functions to use this object.
        *************************************************************************/
        public class rbfmodel
        {
            public int ny;
            public int nx;
            public int nc;
            public int nl;
            public nearestneighbor.kdtree tree;
            public double[,] xc;
            public double[,] wr;
            public double rmax;
            public double[,] v;
            public int gridtype;
            public bool fixrad;
            public double lambdav;
            public double radvalue;
            public double radzvalue;
            public int nlayers;
            public int aterm;
            public int algorithmtype;
            public double epsort;
            public double epserr;
            public int maxits;
            public double h;
            public int n;
            public double[,] x;
            public double[,] y;
            public double[] calcbufxcx;
            public double[,] calcbufx;
            public int[] calcbuftags;
            public rbfmodel()
            {
                tree = new nearestneighbor.kdtree();
                xc = new double[0,0];
                wr = new double[0,0];
                v = new double[0,0];
                x = new double[0,0];
                y = new double[0,0];
                calcbufxcx = new double[0];
                calcbufx = new double[0,0];
                calcbuftags = new int[0];
            }
        };


        /*************************************************************************
        RBF solution report:
        * TerminationType   -   termination type, positive values - success,
                                non-positive - failure.
        *************************************************************************/
        public class rbfreport
        {
            public int arows;
            public int acols;
            public int annz;
            public int iterationscount;
            public int nmv;
            public int terminationtype;
        };




        public const double eps = 1.0E-6;
        public const int mxnx = 3;
        public const double rbffarradius = 6;
        public const double rbfnearradius = 2.1;
        public const double rbfmlradius = 3;
        public const int rbffirstversion = 0;


        /*************************************************************************
        This function creates RBF  model  for  a  scalar (NY=1)  or  vector (NY>1)
        function in a NX-dimensional space (NX=2 or NX=3).

        Newly created model is empty. It can be used for interpolation right after
        creation, but it just returns zeros. You have to add points to the  model,
        tune interpolation settings, and then  call  model  construction  function
        RBFBuildModel() which will update model according to your specification.

        USAGE:
        1. User creates model with RBFCreate()
        2. User adds dataset with RBFSetPoints() (points do NOT have to  be  on  a
           regular grid)
        3. (OPTIONAL) User chooses polynomial term by calling:
           * RBFLinTerm() to set linear term
           * RBFConstTerm() to set constant term
           * RBFZeroTerm() to set zero term
           By default, linear term is used.
        4. User chooses specific RBF algorithm to use: either QNN (RBFSetAlgoQNN)
           or ML (RBFSetAlgoMultiLayer).
        5. User calls RBFBuildModel() function which rebuilds model  according  to
           the specification
        6. User may call RBFCalc() to calculate model value at the specified point,
           RBFGridCalc() to  calculate   model  values at the points of the regular
           grid. User may extract model coefficients with RBFUnpack() call.
           
        INPUT PARAMETERS:
            NX      -   dimension of the space, NX=2 or NX=3
            NY      -   function dimension, NY>=1

        OUTPUT PARAMETERS:
            S       -   RBF model (initially equals to zero)

        NOTE 1: memory requirements. RBF models require amount of memory  which is
                proportional  to  the  number  of data points. Memory is allocated 
                during model construction, but most of this memory is freed  after
                model coefficients are calculated.
                
                Some approximate estimates for N centers with default settings are
                given below:
                * about 250*N*(sizeof(double)+2*sizeof(int)) bytes  of  memory  is
                  needed during model construction stage.
                * about 15*N*sizeof(double) bytes is needed after model is built.
                For example, for N=100000 we may need 0.6 GB of memory  to  build
                model, but just about 0.012 GB to store it.

          -- ALGLIB --
             Copyright 13.12.2011 by Bochkanov Sergey
        *************************************************************************/
        public static void rbfcreate(int nx,
            int ny,
            rbfmodel s)
        {
            int i = 0;
            int j = 0;

            alglib.ap.assert(nx==2 || nx==3, "RBFCreate: NX<>2 and NX<>3");
            alglib.ap.assert(ny>=1, "RBFCreate: NY<1");
            s.nx = nx;
            s.ny = ny;
            s.nl = 0;
            s.nc = 0;
            s.v = new double[ny, mxnx+1];
            for(i=0; i<=ny-1; i++)
            {
                for(j=0; j<=mxnx; j++)
                {
                    s.v[i,j] = 0;
                }
            }
            s.n = 0;
            s.rmax = 0;
            s.gridtype = 2;
            s.fixrad = false;
            s.radvalue = 1;
            s.radzvalue = 5;
            s.aterm = 1;
            s.algorithmtype = 1;
            
            //
            // stopping criteria
            //
            s.epsort = eps;
            s.epserr = eps;
            s.maxits = 0;
        }


        /*************************************************************************
        This function adds dataset.

        This function overrides results of the previous calls, i.e. multiple calls
        of this function will result in only the last set being added.

        INPUT PARAMETERS:
            S       -   RBF model, initialized by RBFCreate() call.
            XY      -   points, array[N,NX+NY]. One row corresponds to  one  point
                        in the dataset. First NX elements  are  coordinates,  next
                        NY elements are function values. Array may  be larger than 
                        specific,  in  this  case  only leading [N,NX+NY] elements 
                        will be used.
            N       -   number of points in the dataset

        After you've added dataset and (optionally) tuned algorithm  settings  you
        should call RBFBuildModel() in order to build a model for you.

        NOTE: this   function  has   some   serialization-related  subtleties.  We
              recommend you to study serialization examples from ALGLIB  Reference
              Manual if you want to perform serialization of your models.
              

          -- ALGLIB --
             Copyright 13.12.2011 by Bochkanov Sergey
        *************************************************************************/
        public static void rbfsetpoints(rbfmodel s,
            double[,] xy,
            int n)
        {
            int i = 0;
            int j = 0;

            alglib.ap.assert(n>0, "RBFSetPoints: N<0");
            alglib.ap.assert(alglib.ap.rows(xy)>=n, "RBFSetPoints: Rows(XY)<N");
            alglib.ap.assert(alglib.ap.cols(xy)>=s.nx+s.ny, "RBFSetPoints: Cols(XY)<NX+NY");
            s.n = n;
            s.x = new double[s.n, mxnx];
            s.y = new double[s.n, s.ny];
            for(i=0; i<=s.n-1; i++)
            {
                for(j=0; j<=mxnx-1; j++)
                {
                    s.x[i,j] = 0;
                }
                for(j=0; j<=s.nx-1; j++)
                {
                    s.x[i,j] = xy[i,j];
                }
                for(j=0; j<=s.ny-1; j++)
                {
                    s.y[i,j] = xy[i,j+s.nx];
                }
            }
        }


        /*************************************************************************
        This  function  sets  RBF interpolation algorithm. ALGLIB supports several
        RBF algorithms with different properties.

        This algorithm is called RBF-QNN and  it  is  good  for  point  sets  with
        following properties:
        a) all points are distinct
        b) all points are well separated.
        c) points  distribution  is  approximately  uniform.  There is no "contour
           lines", clusters of points, or other small-scale structures.

        Algorithm description:
        1) interpolation centers are allocated to data points
        2) interpolation radii are calculated as distances to the  nearest centers
           times Q coefficient (where Q is a value from [0.75,1.50]).
        3) after  performing (2) radii are transformed in order to avoid situation
           when single outlier has very large radius and  influences  many  points
           across all dataset. Transformation has following form:
               new_r[i] = min(r[i],Z*median(r[]))
           where r[i] is I-th radius, median()  is a median  radius across  entire
           dataset, Z is user-specified value which controls amount  of  deviation
           from median radius.

        When (a) is violated,  we  will  be unable to build RBF model. When (b) or
        (c) are violated, model will be built, but interpolation quality  will  be
        low. See http://www.alglib.net/interpolation/ for more information on this
        subject.

        This algorithm is used by default.

        Additional Q parameter controls smoothness properties of the RBF basis:
        * Q<0.75 will give perfectly conditioned basis,  but  terrible  smoothness
          properties (RBF interpolant will have sharp peaks around function values)
        * Q around 1.0 gives good balance between smoothness and condition number
        * Q>1.5 will lead to badly conditioned systems and slow convergence of the
          underlying linear solver (although smoothness will be very good)
        * Q>2.0 will effectively make optimizer useless because it won't  converge
          within reasonable amount of iterations. It is possible to set such large
          Q, but it is advised not to do so.

        INPUT PARAMETERS:
            S       -   RBF model, initialized by RBFCreate() call
            Q       -   Q parameter, Q>0, recommended value - 1.0
            Z       -   Z parameter, Z>0, recommended value - 5.0

        NOTE: this   function  has   some   serialization-related  subtleties.  We
              recommend you to study serialization examples from ALGLIB  Reference
              Manual if you want to perform serialization of your models.


          -- ALGLIB --
             Copyright 13.12.2011 by Bochkanov Sergey
        *************************************************************************/
        public static void rbfsetalgoqnn(rbfmodel s,
            double q,
            double z)
        {
            alglib.ap.assert(math.isfinite(q), "RBFSetAlgoQNN: Q is infinite or NAN");
            alglib.ap.assert((double)(q)>(double)(0), "RBFSetAlgoQNN: Q<=0");
            rbfgridpoints(s);
            rbfradnn(s, q, z);
            s.algorithmtype = 1;
        }


        /*************************************************************************
        This  function  sets  RBF interpolation algorithm. ALGLIB supports several
        RBF algorithms with different properties.

        This  algorithm is called RBF-ML. It builds  multilayer  RBF  model,  i.e.
        model with subsequently decreasing  radii,  which  allows  us  to  combine
        smoothness (due to  large radii of  the first layers) with  exactness (due
        to small radii of the last layers) and fast convergence.

        Internally RBF-ML uses many different  means  of acceleration, from sparse
        matrices  to  KD-trees,  which  results in algorithm whose working time is
        roughly proportional to N*log(N)*Density*RBase^2*NLayers,  where  N  is  a
        number of points, Density is an average density if points per unit of  the
        interpolation space, RBase is an initial radius, NLayers is  a  number  of
        layers.

        RBF-ML is good for following kinds of interpolation problems:
        1. "exact" problems (perfect fit) with well separated points
        2. least squares problems with arbitrary distribution of points (algorithm
           gives  perfect  fit  where it is possible, and resorts to least squares
           fit in the hard areas).
        3. noisy problems where  we  want  to  apply  some  controlled  amount  of
           smoothing.

        INPUT PARAMETERS:
            S       -   RBF model, initialized by RBFCreate() call
            RBase   -   RBase parameter, RBase>0
            NLayers -   NLayers parameter, NLayers>0, recommended value  to  start
                        with - about 5.
            LambdaV -   regularization value, can be useful when  solving  problem
                        in the least squares sense.  Optimal  lambda  is  problem-
                        dependent and require trial and error. In our  experience,
                        good lambda can be as large as 0.1, and you can use  0.001
                        as initial guess.
                        Default  value  - 0.01, which is used when LambdaV is  not
                        given.  You  can  specify  zero  value,  but  it  is   not
                        recommended to do so.

        TUNING ALGORITHM

        In order to use this algorithm you have to choose three parameters:
        * initial radius RBase
        * number of layers in the model NLayers
        * regularization coefficient LambdaV

        Initial radius is easy to choose - you can pick any number  several  times
        larger  than  the  average  distance between points. Algorithm won't break
        down if you choose radius which is too large (model construction time will
        increase, but model will be built correctly).

        Choose such number of layers that RLast=RBase/2^(NLayers-1)  (radius  used
        by  the  last  layer)  will  be  smaller than the typical distance between
        points.  In  case  model  error  is  too large, you can increase number of
        layers.  Having  more  layers  will make model construction and evaluation
        proportionally slower, but it will allow you to have model which precisely
        fits your data. From the other side, if you want to  suppress  noise,  you
        can DECREASE number of layers to make your model less flexible.

        Regularization coefficient LambdaV controls smoothness of  the  individual
        models built for each layer. We recommend you to use default value in case
        you don't want to tune this parameter,  because  having  non-zero  LambdaV
        accelerates and stabilizes internal iterative algorithm. In case you  want
        to suppress noise you can use  LambdaV  as  additional  parameter  (larger
        value = more smoothness) to tune.

        TYPICAL ERRORS

        1. Using  initial  radius  which is too large. Memory requirements  of the
           RBF-ML are roughly proportional to N*Density*RBase^2 (where Density  is
           an average density of points per unit of the interpolation  space).  In
           the extreme case of the very large RBase we will need O(N^2)  units  of
           memory - and many layers in order to decrease radius to some reasonably
           small value.

        2. Using too small number of layers - RBF models with large radius are not
           flexible enough to reproduce small variations in the  target  function.
           You  need  many  layers  with  different radii, from large to small, in
           order to have good model.

        3. Using  initial  radius  which  is  too  small.  You will get model with
           "holes" in the areas which are too far away from interpolation centers.
           However, algorithm will work correctly (and quickly) in this case.

        4. Using too many layers - you will get too large and too slow model. This
           model  will  perfectly  reproduce  your function, but maybe you will be
           able to achieve similar results with less layers (and less memory).
           
          -- ALGLIB --
             Copyright 02.03.2012 by Bochkanov Sergey
        *************************************************************************/
        public static void rbfsetalgomultilayer(rbfmodel s,
            double rbase,
            int nlayers,
            double lambdav)
        {
            alglib.ap.assert(math.isfinite(rbase), "RBFSetAlgoMultiLayer: RBase is infinite or NaN");
            alglib.ap.assert((double)(rbase)>(double)(0), "RBFSetAlgoMultiLayer: RBase<=0");
            alglib.ap.assert(nlayers>=0, "RBFSetAlgoMultiLayer: NLayers<0");
            alglib.ap.assert(math.isfinite(lambdav), "RBFSetAlgoMultiLayer: LambdaV is infinite or NAN");
            alglib.ap.assert((double)(lambdav)>=(double)(0), "RBFSetAlgoMultiLayer: LambdaV<0");
            s.radvalue = rbase;
            s.nlayers = nlayers;
            s.algorithmtype = 2;
            s.lambdav = lambdav;
        }


        /*************************************************************************
        This function sets linear term (model is a sum of radial  basis  functions
        plus linear polynomial). This function won't have effect until  next  call 
        to RBFBuildModel().

        INPUT PARAMETERS:
            S       -   RBF model, initialized by RBFCreate() call

        NOTE: this   function  has   some   serialization-related  subtleties.  We
              recommend you to study serialization examples from ALGLIB  Reference
              Manual if you want to perform serialization of your models.

          -- ALGLIB --
             Copyright 13.12.2011 by Bochkanov Sergey
        *************************************************************************/
        public static void rbfsetlinterm(rbfmodel s)
        {
            s.aterm = 1;
        }


        /*************************************************************************
        This function sets constant term (model is a sum of radial basis functions
        plus constant).  This  function  won't  have  effect  until  next  call to 
        RBFBuildModel().

        INPUT PARAMETERS:
            S       -   RBF model, initialized by RBFCreate() call

        NOTE: this   function  has   some   serialization-related  subtleties.  We
              recommend you to study serialization examples from ALGLIB  Reference
              Manual if you want to perform serialization of your models.

          -- ALGLIB --
             Copyright 13.12.2011 by Bochkanov Sergey
        *************************************************************************/
        public static void rbfsetconstterm(rbfmodel s)
        {
            s.aterm = 2;
        }


        /*************************************************************************
        This  function  sets  zero  term (model is a sum of radial basis functions 
        without polynomial term). This function won't have effect until next  call
        to RBFBuildModel().

        INPUT PARAMETERS:
            S       -   RBF model, initialized by RBFCreate() call

        NOTE: this   function  has   some   serialization-related  subtleties.  We
              recommend you to study serialization examples from ALGLIB  Reference
              Manual if you want to perform serialization of your models.

          -- ALGLIB --
             Copyright 13.12.2011 by Bochkanov Sergey
        *************************************************************************/
        public static void rbfsetzeroterm(rbfmodel s)
        {
            s.aterm = 3;
        }


        /*************************************************************************
        This function sets stopping criteria of the underlying linear solver.

        INPUT PARAMETERS:
            S       -   RBF model, initialized by RBFCreate() call
            EpsOrt  -   orthogonality stopping criterion, EpsOrt>=0. Algorithm will
                        stop when ||A'*r||<=EpsOrt where A' is a transpose of  the 
                        system matrix, r is a residual vector.
                        Recommended value of EpsOrt is equal to 1E-6.
                        This criterion will stop algorithm when we have "bad fit"
                        situation, i.e. when we should stop in a point with large,
                        nonzero residual.
            EpsErr  -   residual stopping  criterion.  Algorithm  will  stop  when
                        ||r||<=EpsErr*||b||, where r is a residual vector, b is  a
                        right part of the system (function values).
                        Recommended value of EpsErr is equal to 1E-3 or 1E-6.
                        This  criterion  will  stop  algorithm  in  a  "good  fit" 
                        situation when we have near-zero residual near the desired
                        solution.
            MaxIts  -   this criterion will stop algorithm after MaxIts iterations.
                        It should be used for debugging purposes only!
                        Zero MaxIts means that no limit is placed on the number of
                        iterations.

        We  recommend  to  set  moderate  non-zero  values   EpsOrt   and   EpsErr 
        simultaneously. Values equal to 10E-6 are good to start with. In case  you
        need high performance and do not need high precision ,  you  may  decrease
        EpsErr down to 0.001. However, we do not recommend decreasing EpsOrt.

        As for MaxIts, we recommend to leave it zero unless you know what you do.

        NOTE: this   function  has   some   serialization-related  subtleties.  We
              recommend you to study serialization examples from ALGLIB  Reference
              Manual if you want to perform serialization of your models.

          -- ALGLIB --
             Copyright 13.12.2011 by Bochkanov Sergey
        *************************************************************************/
        public static void rbfsetcond(rbfmodel s,
            double epsort,
            double epserr,
            int maxits)
        {
            alglib.ap.assert(math.isfinite(epsort) && (double)(epsort)>=(double)(0), "RBFSetCond: EpsOrt is negative, INF or NAN");
            alglib.ap.assert(math.isfinite(epserr) && (double)(epserr)>=(double)(0), "RBFSetCond: EpsB is negative, INF or NAN");
            alglib.ap.assert(maxits>=0, "RBFSetCond: MaxIts is negative");
            if( ((double)(epsort)==(double)(0) && (double)(epserr)==(double)(0)) && maxits==0 )
            {
                s.epsort = eps;
                s.epserr = eps;
                s.maxits = 0;
            }
            else
            {
                s.epsort = epsort;
                s.epserr = epserr;
                s.maxits = maxits;
            }
        }


        /*************************************************************************
        This   function  builds  RBF  model  and  returns  report  (contains  some 
        information which can be used for evaluation of the algorithm properties).

        Call to this function modifies RBF model by calculating its centers/radii/
        weights  and  saving  them  into  RBFModel  structure.  Initially RBFModel 
        contain zero coefficients, but after call to this function  we  will  have
        coefficients which were calculated in order to fit our dataset.

        After you called this function you can call RBFCalc(),  RBFGridCalc()  and
        other model calculation functions.

        INPUT PARAMETERS:
            S       -   RBF model, initialized by RBFCreate() call
            Rep     -   report:
                        * Rep.TerminationType:
                          * -5 - non-distinct basis function centers were detected,
                                 interpolation aborted
                          * -4 - nonconvergence of the internal SVD solver
                          *  1 - successful termination
                        Fields are used for debugging purposes:
                        * Rep.IterationsCount - iterations count of the LSQR solver
                        * Rep.NMV - number of matrix-vector products
                        * Rep.ARows - rows count for the system matrix
                        * Rep.ACols - columns count for the system matrix
                        * Rep.ANNZ - number of significantly non-zero elements
                          (elements above some algorithm-determined threshold)

        NOTE:  failure  to  build  model will leave current state of the structure
        unchanged.

          -- ALGLIB --
             Copyright 13.12.2011 by Bochkanov Sergey
        *************************************************************************/
        public static void rbfbuildmodel(rbfmodel s,
            rbfreport rep)
        {
            nearestneighbor.kdtree tree = new nearestneighbor.kdtree();
            nearestneighbor.kdtree ctree = new nearestneighbor.kdtree();
            double[] dist = new double[0];
            double[] xcx = new double[0];
            double[,] a = new double[0,0];
            double[,] v = new double[0,0];
            double[,] omega = new double[0,0];
            double[] y = new double[0];
            double[,] residualy = new double[0,0];
            double[] radius = new double[0];
            double[,] xc = new double[0,0];
            double[] mnx = new double[0];
            double[] mxx = new double[0];
            double[] edge = new double[0];
            int[] mxsteps = new int[0];
            int nc = 0;
            double rmax = 0;
            int[] tags = new int[0];
            int[] ctags = new int[0];
            int i = 0;
            int j = 0;
            int k = 0;
            int k2 = 0;
            int snnz = 0;
            double[] tmp0 = new double[0];
            double[] tmp1 = new double[0];
            int layerscnt = 0;

            alglib.ap.assert(s.nx==2 || s.nx==3, "RBFBuildModel: S.NX<>2 or S.NX<>3!");
            
            //
            // Quick exit when we have no points or if setted multilayer algorithm
            // by user without layers.
            //
            if( s.n==0 || (s.algorithmtype==2 && s.nlayers==0) )
            {
                rep.terminationtype = 1;
                rep.iterationscount = 0;
                rep.nmv = 0;
                rep.arows = 0;
                rep.acols = 0;
                nearestneighbor.kdtreebuildtagged(s.xc, tags, 0, mxnx, 0, 2, s.tree);
                s.xc = new double[0, 0];
                s.wr = new double[0, 0];
                s.nc = 0;
                s.rmax = 0;
                s.v = new double[s.ny, mxnx+1];
                for(i=0; i<=s.ny-1; i++)
                {
                    for(j=0; j<=mxnx; j++)
                    {
                        s.v[i,j] = 0;
                    }
                }
                return;
            }
            
            //
            // General case, N>0
            //
            rep.annz = 0;
            rep.iterationscount = 0;
            rep.nmv = 0;
            xcx = new double[mxnx];
            
            //
            // First model in a sequence - linear model.
            // Residuals from linear regression are stored in the ResidualY variable
            // (used later to build RBF models).
            //
            residualy = new double[s.n, s.ny];
            for(i=0; i<=s.n-1; i++)
            {
                for(j=0; j<=s.ny-1; j++)
                {
                    residualy[i,j] = s.y[i,j];
                }
            }
            if( !buildlinearmodel(s.x, ref residualy, s.n, s.ny, s.aterm, ref v) )
            {
                rep.terminationtype = -5;
                return;
            }
            
            //
            // Second model in a sequence - RBF term.
            //
            // NOTE: assignments below are not necessary, but without them
            //       MSVC complains about unitialized variables.
            //
            nc = 0;
            rmax = 0;
            layerscnt = 0;
            if( s.algorithmtype==1 )
            {
                
                //
                // Add RBF model.
                // This model uses local KD-trees to speed-up nearest neighbor searches.
                //
                if( s.gridtype==1 )
                {
                    mxx = new double[s.nx];
                    mnx = new double[s.nx];
                    mxsteps = new int[s.nx];
                    edge = new double[s.nx];
                    for(i=0; i<=s.nx-1; i++)
                    {
                        mxx[i] = s.x[0,i];
                        mnx[i] = s.x[0,i];
                    }
                    for(i=0; i<=s.n-1; i++)
                    {
                        for(j=0; j<=s.nx-1; j++)
                        {
                            if( (double)(mxx[j])<(double)(s.x[i,j]) )
                            {
                                mxx[j] = s.x[i,j];
                            }
                            if( (double)(mnx[j])>(double)(s.x[i,j]) )
                            {
                                mnx[j] = s.x[i,j];
                            }
                        }
                    }
                    for(i=0; i<=s.nx-1; i++)
                    {
                        mxsteps[i] = (int)((mxx[i]-mnx[i])/(2*s.h))+1;
                        edge[i] = (mxx[i]+mnx[i])/2-s.h*mxsteps[i];
                    }
                    nc = 1;
                    for(i=0; i<=s.nx-1; i++)
                    {
                        mxsteps[i] = 2*mxsteps[i]+1;
                        nc = nc*mxsteps[i];
                    }
                    xc = new double[nc, mxnx];
                    if( s.nx==2 )
                    {
                        for(i=0; i<=mxsteps[0]-1; i++)
                        {
                            for(j=0; j<=mxsteps[1]-1; j++)
                            {
                                for(k2=0; k2<=mxnx-1; k2++)
                                {
                                    xc[i*mxsteps[1]+j,k2] = 0;
                                }
                                xc[i*mxsteps[1]+j,0] = edge[0]+s.h*i;
                                xc[i*mxsteps[1]+j,1] = edge[1]+s.h*j;
                            }
                        }
                    }
                    if( s.nx==3 )
                    {
                        for(i=0; i<=mxsteps[0]-1; i++)
                        {
                            for(j=0; j<=mxsteps[1]-1; j++)
                            {
                                for(k=0; k<=mxsteps[2]-1; k++)
                                {
                                    for(k2=0; k2<=mxnx-1; k2++)
                                    {
                                        xc[i*mxsteps[1]+j,k2] = 0;
                                    }
                                    xc[(i*mxsteps[1]+j)*mxsteps[2]+k,0] = edge[0]+s.h*i;
                                    xc[(i*mxsteps[1]+j)*mxsteps[2]+k,1] = edge[1]+s.h*j;
                                    xc[(i*mxsteps[1]+j)*mxsteps[2]+k,2] = edge[2]+s.h*k;
                                }
                            }
                        }
                    }
                }
                else
                {
                    if( s.gridtype==2 )
                    {
                        nc = s.n;
                        xc = new double[nc, mxnx];
                        for(i=0; i<=nc-1; i++)
                        {
                            for(j=0; j<=mxnx-1; j++)
                            {
                                xc[i,j] = s.x[i,j];
                            }
                        }
                    }
                    else
                    {
                        if( s.gridtype==3 )
                        {
                            nc = s.nc;
                            xc = new double[nc, mxnx];
                            for(i=0; i<=nc-1; i++)
                            {
                                for(j=0; j<=mxnx-1; j++)
                                {
                                    xc[i,j] = s.xc[i,j];
                                }
                            }
                        }
                        else
                        {
                            alglib.ap.assert(false, "RBFBuildModel: either S.GridType<1 or S.GridType>3!");
                        }
                    }
                }
                rmax = 0;
                radius = new double[nc];
                ctags = new int[nc];
                for(i=0; i<=nc-1; i++)
                {
                    ctags[i] = i;
                }
                nearestneighbor.kdtreebuildtagged(xc, ctags, nc, mxnx, 0, 2, ctree);
                if( s.fixrad )
                {
                    
                    //
                    // Fixed radius
                    //
                    for(i=0; i<=nc-1; i++)
                    {
                        radius[i] = s.radvalue;
                    }
                    rmax = radius[0];
                }
                else
                {
                    
                    //
                    // Dynamic radius
                    //
                    if( nc==0 )
                    {
                        rmax = 1;
                    }
                    else
                    {
                        if( nc==1 )
                        {
                            radius[0] = s.radvalue;
                            rmax = radius[0];
                        }
                        else
                        {
                            
                            //
                            // NC>1, calculate radii using distances to nearest neigbors
                            //
                            for(i=0; i<=nc-1; i++)
                            {
                                for(j=0; j<=mxnx-1; j++)
                                {
                                    xcx[j] = xc[i,j];
                                }
                                if( nearestneighbor.kdtreequeryknn(ctree, xcx, 1, false)>0 )
                                {
                                    nearestneighbor.kdtreequeryresultsdistances(ctree, ref dist);
                                    radius[i] = s.radvalue*dist[0];
                                }
                                else
                                {
                                    
                                    //
                                    // No neighbors found (it will happen when we have only one center).
                                    // Initialize radius with default value.
                                    //
                                    radius[i] = 1.0;
                                }
                            }
                            
                            //
                            // Apply filtering
                            //
                            apserv.rvectorsetlengthatleast(ref tmp0, nc);
                            for(i=0; i<=nc-1; i++)
                            {
                                tmp0[i] = radius[i];
                            }
                            tsort.tagsortfast(ref tmp0, ref tmp1, nc);
                            for(i=0; i<=nc-1; i++)
                            {
                                radius[i] = Math.Min(radius[i], s.radzvalue*tmp0[nc/2]);
                            }
                            
                            //
                            // Calculate RMax, check that all radii are non-zero
                            //
                            for(i=0; i<=nc-1; i++)
                            {
                                rmax = Math.Max(rmax, radius[i]);
                            }
                            for(i=0; i<=nc-1; i++)
                            {
                                if( (double)(radius[i])==(double)(0) )
                                {
                                    rep.terminationtype = -5;
                                    return;
                                }
                            }
                        }
                    }
                }
                apserv.ivectorsetlengthatleast(ref tags, s.n);
                for(i=0; i<=s.n-1; i++)
                {
                    tags[i] = i;
                }
                nearestneighbor.kdtreebuildtagged(s.x, tags, s.n, mxnx, 0, 2, tree);
                buildrbfmodellsqr(s.x, ref residualy, xc, radius, s.n, nc, s.ny, tree, ctree, s.epsort, s.epserr, s.maxits, ref rep.annz, ref snnz, ref omega, ref rep.terminationtype, ref rep.iterationscount, ref rep.nmv);
                layerscnt = 1;
            }
            else
            {
                if( s.algorithmtype==2 )
                {
                    rmax = s.radvalue;
                    buildrbfmlayersmodellsqr(s.x, ref residualy, ref xc, s.radvalue, ref radius, s.n, ref nc, s.ny, s.nlayers, ctree, 1.0E-6, 1.0E-6, 50, s.lambdav, ref rep.annz, ref omega, ref rep.terminationtype, ref rep.iterationscount, ref rep.nmv);
                    layerscnt = s.nlayers;
                }
                else
                {
                    alglib.ap.assert(false, "RBFBuildModel: internal error(AlgorithmType neither 1 nor 2)");
                }
            }
            if( rep.terminationtype<=0 )
            {
                return;
            }
            
            //
            // Model is built
            //
            s.nc = nc/layerscnt;
            s.rmax = rmax;
            s.nl = layerscnt;
            s.xc = new double[s.nc, mxnx];
            s.wr = new double[s.nc, 1+s.nl*s.ny];
            s.v = new double[s.ny, mxnx+1];
            for(i=0; i<=s.nc-1; i++)
            {
                for(j=0; j<=mxnx-1; j++)
                {
                    s.xc[i,j] = xc[i,j];
                }
            }
            apserv.ivectorsetlengthatleast(ref tags, s.nc);
            for(i=0; i<=s.nc-1; i++)
            {
                tags[i] = i;
            }
            nearestneighbor.kdtreebuildtagged(s.xc, tags, s.nc, mxnx, 0, 2, s.tree);
            for(i=0; i<=s.nc-1; i++)
            {
                s.wr[i,0] = radius[i];
                for(k=0; k<=layerscnt-1; k++)
                {
                    for(j=0; j<=s.ny-1; j++)
                    {
                        s.wr[i,1+k*s.ny+j] = omega[k*s.nc+i,j];
                    }
                }
            }
            for(i=0; i<=s.ny-1; i++)
            {
                for(j=0; j<=mxnx; j++)
                {
                    s.v[i,j] = v[i,j];
                }
            }
            rep.terminationtype = 1;
            rep.arows = s.n;
            rep.acols = s.nc;
        }


        /*************************************************************************
        This function calculates values of the RBF model in the given point.

        This function should be used when we have NY=1 (scalar function) and  NX=2
        (2-dimensional space). If you have 3-dimensional space, use RBFCalc3(). If
        you have general situation (NX-dimensional space, NY-dimensional function)
        you should use general, less efficient implementation RBFCalc().

        If  you  want  to  calculate  function  values  many times, consider using 
        RBFGridCalc2(), which is far more efficient than many subsequent calls  to
        RBFCalc2().

        This function returns 0.0 when:
        * model is not initialized
        * NX<>2
         *NY<>1

        INPUT PARAMETERS:
            S       -   RBF model
            X0      -   first coordinate, finite number
            X1      -   second coordinate, finite number

        RESULT:
            value of the model or 0.0 (as defined above)

          -- ALGLIB --
             Copyright 13.12.2011 by Bochkanov Sergey
        *************************************************************************/
        public static double rbfcalc2(rbfmodel s,
            double x0,
            double x1)
        {
            double result = 0;
            int i = 0;
            int j = 0;
            int lx = 0;
            int tg = 0;
            double d2 = 0;
            double t = 0;
            double bfcur = 0;
            double rcur = 0;

            alglib.ap.assert(math.isfinite(x0), "RBFCalc2: invalid value for X0 (X0 is Inf)!");
            alglib.ap.assert(math.isfinite(x1), "RBFCalc2: invalid value for X1 (X1 is Inf)!");
            result = 0;
            if( (s.ny!=1 || s.nx!=2) || s.nc==0 )
            {
                return result;
            }
            apserv.rvectorsetlengthatleast(ref s.calcbufxcx, mxnx);
            for(i=0; i<=mxnx-1; i++)
            {
                s.calcbufxcx[i] = 0.0;
            }
            s.calcbufxcx[0] = x0;
            s.calcbufxcx[1] = x1;
            lx = nearestneighbor.kdtreequeryrnn(s.tree, s.calcbufxcx, s.rmax*rbffarradius, true);
            nearestneighbor.kdtreequeryresultsx(s.tree, ref s.calcbufx);
            nearestneighbor.kdtreequeryresultstags(s.tree, ref s.calcbuftags);
            for(i=0; i<=lx-1; i++)
            {
                tg = s.calcbuftags[i];
                d2 = math.sqr(x0-s.calcbufx[i,0])+math.sqr(x1-s.calcbufx[i,1]);
                rcur = s.wr[tg,0];
                bfcur = Math.Exp(-(d2/(rcur*rcur)));
                for(j=0; j<=s.nl-1; j++)
                {
                    result = result+bfcur*s.wr[tg,1+j];
                    rcur = 0.5*rcur;
                    t = bfcur*bfcur;
                    bfcur = t*t;
                }
            }
            result = result+s.v[0,0]*x0+s.v[0,1]*x1+s.v[0,mxnx];
            return result;
        }


        /*************************************************************************
        This function calculates values of the RBF model in the given point.

        This function should be used when we have NY=1 (scalar function) and  NX=3
        (3-dimensional space). If you have 2-dimensional space, use RBFCalc2(). If
        you have general situation (NX-dimensional space, NY-dimensional function)
        you should use general, less efficient implementation RBFCalc().

        This function returns 0.0 when:
        * model is not initialized
        * NX<>3
         *NY<>1

        INPUT PARAMETERS:
            S       -   RBF model
            X0      -   first coordinate, finite number
            X1      -   second coordinate, finite number
            X2      -   third coordinate, finite number

        RESULT:
            value of the model or 0.0 (as defined above)

          -- ALGLIB --
             Copyright 13.12.2011 by Bochkanov Sergey
        *************************************************************************/
        public static double rbfcalc3(rbfmodel s,
            double x0,
            double x1,
            double x2)
        {
            double result = 0;
            int i = 0;
            int j = 0;
            int lx = 0;
            int tg = 0;
            double t = 0;
            double rcur = 0;
            double bf = 0;

            alglib.ap.assert(math.isfinite(x0), "RBFCalc3: invalid value for X0 (X0 is Inf or NaN)!");
            alglib.ap.assert(math.isfinite(x1), "RBFCalc3: invalid value for X1 (X1 is Inf or NaN)!");
            alglib.ap.assert(math.isfinite(x2), "RBFCalc3: invalid value for X2 (X2 is Inf or NaN)!");
            result = 0;
            if( (s.ny!=1 || s.nx!=3) || s.nc==0 )
            {
                return result;
            }
            
            //
            // calculating value for F(X)
            //
            apserv.rvectorsetlengthatleast(ref s.calcbufxcx, mxnx);
            for(i=0; i<=mxnx-1; i++)
            {
                s.calcbufxcx[i] = 0.0;
            }
            s.calcbufxcx[0] = x0;
            s.calcbufxcx[1] = x1;
            s.calcbufxcx[2] = x2;
            lx = nearestneighbor.kdtreequeryrnn(s.tree, s.calcbufxcx, s.rmax*rbffarradius, true);
            nearestneighbor.kdtreequeryresultsx(s.tree, ref s.calcbufx);
            nearestneighbor.kdtreequeryresultstags(s.tree, ref s.calcbuftags);
            for(i=0; i<=lx-1; i++)
            {
                tg = s.calcbuftags[i];
                rcur = s.wr[tg,0];
                bf = Math.Exp(-((math.sqr(x0-s.calcbufx[i,0])+math.sqr(x1-s.calcbufx[i,1])+math.sqr(x2-s.calcbufx[i,2]))/math.sqr(rcur)));
                for(j=0; j<=s.nl-1; j++)
                {
                    result = result+bf*s.wr[tg,1+j];
                    t = bf*bf;
                    bf = t*t;
                }
            }
            result = result+s.v[0,0]*x0+s.v[0,1]*x1+s.v[0,2]*x2+s.v[0,mxnx];
            return result;
        }


        /*************************************************************************
        This function calculates values of the RBF model at the given point.

        This is general function which can be used for arbitrary NX (dimension  of 
        the space of arguments) and NY (dimension of the function itself). However
        when  you  have  NY=1  you  may  find more convenient to use RBFCalc2() or 
        RBFCalc3().

        This function returns 0.0 when model is not initialized.

        INPUT PARAMETERS:
            S       -   RBF model
            X       -   coordinates, array[NX].
                        X may have more than NX elements, in this case only 
                        leading NX will be used.

        OUTPUT PARAMETERS:
            Y       -   function value, array[NY]. Y is out-parameter and 
                        reallocated after call to this function. In case you  want
                        to reuse previously allocated Y, you may use RBFCalcBuf(),
                        which reallocates Y only when it is too small.

          -- ALGLIB --
             Copyright 13.12.2011 by Bochkanov Sergey
        *************************************************************************/
        public static void rbfcalc(rbfmodel s,
            double[] x,
            ref double[] y)
        {
            y = new double[0];

            alglib.ap.assert(alglib.ap.len(x)>=s.nx, "RBFCalc: Length(X)<NX");
            alglib.ap.assert(apserv.isfinitevector(x, s.nx), "RBFCalc: X contains infinite or NaN values");
            rbfcalcbuf(s, x, ref y);
        }


        /*************************************************************************
        This function calculates values of the RBF model at the given point.

        Same as RBFCalc(), but does not reallocate Y when in is large enough to 
        store function values.

        INPUT PARAMETERS:
            S       -   RBF model
            X       -   coordinates, array[NX].
                        X may have more than NX elements, in this case only 
                        leading NX will be used.
            Y       -   possibly preallocated array

        OUTPUT PARAMETERS:
            Y       -   function value, array[NY]. Y is not reallocated when it
                        is larger than NY.

          -- ALGLIB --
             Copyright 13.12.2011 by Bochkanov Sergey
        *************************************************************************/
        public static void rbfcalcbuf(rbfmodel s,
            double[] x,
            ref double[] y)
        {
            int i = 0;
            int j = 0;
            int k = 0;
            int lx = 0;
            int tg = 0;
            double t = 0;
            double rcur = 0;
            double bf = 0;

            alglib.ap.assert(alglib.ap.len(x)>=s.nx, "RBFCalcBuf: Length(X)<NX");
            alglib.ap.assert(apserv.isfinitevector(x, s.nx), "RBFCalcBuf: X contains infinite or NaN values");
            if( alglib.ap.len(y)<s.ny )
            {
                y = new double[s.ny];
            }
            for(i=0; i<=s.ny-1; i++)
            {
                y[i] = 0;
            }
            if( s.nc==0 )
            {
                return;
            }
            s.calcbufxcx = new double[mxnx];
            for(i=0; i<=mxnx-1; i++)
            {
                s.calcbufxcx[i] = 0.0;
            }
            for(i=0; i<=s.nx-1; i++)
            {
                s.calcbufxcx[i] = x[i];
            }
            lx = nearestneighbor.kdtreequeryrnn(s.tree, s.calcbufxcx, s.rmax*rbffarradius, true);
            nearestneighbor.kdtreequeryresultsx(s.tree, ref s.calcbufx);
            nearestneighbor.kdtreequeryresultstags(s.tree, ref s.calcbuftags);
            for(i=0; i<=s.ny-1; i++)
            {
                for(j=0; j<=lx-1; j++)
                {
                    tg = s.calcbuftags[j];
                    rcur = s.wr[tg,0];
                    bf = Math.Exp(-((math.sqr(s.calcbufxcx[0]-s.calcbufx[j,0])+math.sqr(s.calcbufxcx[1]-s.calcbufx[j,1])+math.sqr(s.calcbufxcx[2]-s.calcbufx[j,2]))/math.sqr(rcur)));
                    for(k=0; k<=s.nl-1; k++)
                    {
                        y[i] = y[i]+bf*s.wr[tg,1+k*s.ny+i];
                        t = bf*bf;
                        bf = t*t;
                    }
                }
                y[i] = y[i]+s.v[i,0]*s.calcbufxcx[0]+s.v[i,1]*s.calcbufxcx[1]+s.v[i,2]*s.calcbufxcx[2]+s.v[i,mxnx];
            }
        }


        /*************************************************************************
        This function calculates values of the RBF model at the regular grid.

        Grid have N0*N1 points, with Point[I,J] = (X0[I], X1[J])

        This function returns 0.0 when:
        * model is not initialized
        * NX<>2
         *NY<>1

        INPUT PARAMETERS:
            S       -   RBF model
            X0      -   array of grid nodes, first coordinates, array[N0]
            N0      -   grid size (number of nodes) in the first dimension
            X1      -   array of grid nodes, second coordinates, array[N1]
            N1      -   grid size (number of nodes) in the second dimension

        OUTPUT PARAMETERS:
            Y       -   function values, array[N0,N1]. Y is out-variable and 
                        is reallocated by this function.

          -- ALGLIB --
             Copyright 13.12.2011 by Bochkanov Sergey
        *************************************************************************/
        public static void rbfgridcalc2(rbfmodel s,
            double[] x0,
            int n0,
            double[] x1,
            int n1,
            ref double[,] y)
        {
            double[] cpx0 = new double[0];
            double[] cpx1 = new double[0];
            int[] p01 = new int[0];
            int[] p11 = new int[0];
            int[] p2 = new int[0];
            double rlimit = 0;
            double xcnorm2 = 0;
            int hp01 = 0;
            double hcpx0 = 0;
            double xc0 = 0;
            double xc1 = 0;
            double omega = 0;
            double radius = 0;
            int i = 0;
            int j = 0;
            int k = 0;
            int d = 0;
            int i00 = 0;
            int i01 = 0;
            int i10 = 0;
            int i11 = 0;

            y = new double[0,0];

            alglib.ap.assert(n0>0, "RBFGridCalc2: invalid value for N0 (N0<=0)!");
            alglib.ap.assert(n1>0, "RBFGridCalc2: invalid value for N1 (N1<=0)!");
            alglib.ap.assert(alglib.ap.len(x0)>=n0, "RBFGridCalc2: Length(X0)<N0");
            alglib.ap.assert(alglib.ap.len(x1)>=n1, "RBFGridCalc2: Length(X1)<N1");
            alglib.ap.assert(apserv.isfinitevector(x0, n0), "RBFGridCalc2: X0 contains infinite or NaN values!");
            alglib.ap.assert(apserv.isfinitevector(x1, n1), "RBFGridCalc2: X1 contains infinite or NaN values!");
            y = new double[n0, n1];
            for(i=0; i<=n0-1; i++)
            {
                for(j=0; j<=n1-1; j++)
                {
                    y[i,j] = 0;
                }
            }
            if( (s.ny!=1 || s.nx!=2) || s.nc==0 )
            {
                return;
            }
            
            //
            //create and sort arrays
            //
            cpx0 = new double[n0];
            for(i=0; i<=n0-1; i++)
            {
                cpx0[i] = x0[i];
            }
            tsort.tagsort(ref cpx0, n0, ref p01, ref p2);
            cpx1 = new double[n1];
            for(i=0; i<=n1-1; i++)
            {
                cpx1[i] = x1[i];
            }
            tsort.tagsort(ref cpx1, n1, ref p11, ref p2);
            
            //
            //calculate function's value
            //
            for(i=0; i<=s.nc-1; i++)
            {
                radius = s.wr[i,0];
                for(d=0; d<=s.nl-1; d++)
                {
                    omega = s.wr[i,1+d];
                    rlimit = radius*rbffarradius;
                    
                    //
                    //search lower and upper indexes
                    //
                    i00 = tsort.lowerbound(cpx0, n0, s.xc[i,0]-rlimit);
                    i01 = tsort.upperbound(cpx0, n0, s.xc[i,0]+rlimit);
                    i10 = tsort.lowerbound(cpx1, n1, s.xc[i,1]-rlimit);
                    i11 = tsort.upperbound(cpx1, n1, s.xc[i,1]+rlimit);
                    xc0 = s.xc[i,0];
                    xc1 = s.xc[i,1];
                    for(j=i00; j<=i01-1; j++)
                    {
                        hcpx0 = cpx0[j];
                        hp01 = p01[j];
                        for(k=i10; k<=i11-1; k++)
                        {
                            xcnorm2 = math.sqr(hcpx0-xc0)+math.sqr(cpx1[k]-xc1);
                            if( (double)(xcnorm2)<=(double)(rlimit*rlimit) )
                            {
                                y[hp01,p11[k]] = y[hp01,p11[k]]+Math.Exp(-(xcnorm2/math.sqr(radius)))*omega;
                            }
                        }
                    }
                    radius = 0.5*radius;
                }
            }
            
            //
            //add linear term
            //
            for(i=0; i<=n0-1; i++)
            {
                for(j=0; j<=n1-1; j++)
                {
                    y[i,j] = y[i,j]+s.v[0,0]*x0[i]+s.v[0,1]*x1[j]+s.v[0,mxnx];
                }
            }
        }


        /*************************************************************************
        This function "unpacks" RBF model by extracting its coefficients.

        INPUT PARAMETERS:
            S       -   RBF model

        OUTPUT PARAMETERS:
            NX      -   dimensionality of argument
            NY      -   dimensionality of the target function
            XWR     -   model information, array[NC,NX+NY+1].
                        One row of the array corresponds to one basis function:
                        * first NX columns  - coordinates of the center 
                        * next NY columns   - weights, one per dimension of the 
                                              function being modelled
                        * last column       - radius, same for all dimensions of
                                              the function being modelled
            NC      -   number of the centers
            V       -   polynomial  term , array[NY,NX+1]. One row per one 
                        dimension of the function being modelled. First NX 
                        elements are linear coefficients, V[NX] is equal to the 
                        constant part.

          -- ALGLIB --
             Copyright 13.12.2011 by Bochkanov Sergey
        *************************************************************************/
        public static void rbfunpack(rbfmodel s,
            ref int nx,
            ref int ny,
            ref double[,] xwr,
            ref int nc,
            ref double[,] v)
        {
            int i = 0;
            int j = 0;
            double rcur = 0;
            int i_ = 0;
            int i1_ = 0;

            nx = 0;
            ny = 0;
            xwr = new double[0,0];
            nc = 0;
            v = new double[0,0];

            nx = s.nx;
            ny = s.ny;
            nc = s.nc;
            
            //
            // Fill V
            //
            v = new double[s.ny, s.nx+1];
            for(i=0; i<=s.ny-1; i++)
            {
                for(i_=0; i_<=s.nx-1;i_++)
                {
                    v[i,i_] = s.v[i,i_];
                }
                v[i,s.nx] = s.v[i,mxnx];
            }
            
            //
            // Fill XWR and V
            //
            if( nc*s.nl>0 )
            {
                xwr = new double[s.nc*s.nl, s.nx+s.ny+1];
                for(i=0; i<=s.nc-1; i++)
                {
                    rcur = s.wr[i,0];
                    for(j=0; j<=s.nl-1; j++)
                    {
                        for(i_=0; i_<=s.nx-1;i_++)
                        {
                            xwr[i*s.nl+j,i_] = s.xc[i,i_];
                        }
                        i1_ = (1+j*s.ny) - (s.nx);
                        for(i_=s.nx; i_<=s.nx+s.ny-1;i_++)
                        {
                            xwr[i*s.nl+j,i_] = s.wr[i,i_+i1_];
                        }
                        xwr[i*s.nl+j,s.nx+s.ny] = rcur;
                        rcur = 0.5*rcur;
                    }
                }
            }
        }


        /*************************************************************************
        Serializer: allocation

          -- ALGLIB --
             Copyright 02.02.2012 by Bochkanov Sergey
        *************************************************************************/
        public static void rbfalloc(alglib.serializer s,
            rbfmodel model)
        {
            
            //
            // Header
            //
            s.alloc_entry();
            s.alloc_entry();
            
            //
            // Data
            //
            s.alloc_entry();
            s.alloc_entry();
            s.alloc_entry();
            s.alloc_entry();
            nearestneighbor.kdtreealloc(s, model.tree);
            apserv.allocrealmatrix(s, model.xc, -1, -1);
            apserv.allocrealmatrix(s, model.wr, -1, -1);
            s.alloc_entry();
            apserv.allocrealmatrix(s, model.v, -1, -1);
        }


        /*************************************************************************
        Serializer: serialization

          -- ALGLIB --
             Copyright 02.02.2012 by Bochkanov Sergey
        *************************************************************************/
        public static void rbfserialize(alglib.serializer s,
            rbfmodel model)
        {
            
            //
            // Header
            //
            s.serialize_int(scodes.getrbfserializationcode());
            s.serialize_int(rbffirstversion);
            
            //
            // Data
            //
            s.serialize_int(model.nx);
            s.serialize_int(model.ny);
            s.serialize_int(model.nc);
            s.serialize_int(model.nl);
            nearestneighbor.kdtreeserialize(s, model.tree);
            apserv.serializerealmatrix(s, model.xc, -1, -1);
            apserv.serializerealmatrix(s, model.wr, -1, -1);
            s.serialize_double(model.rmax);
            apserv.serializerealmatrix(s, model.v, -1, -1);
        }


        /*************************************************************************
        Serializer: unserialization

          -- ALGLIB --
             Copyright 02.02.2012 by Bochkanov Sergey
        *************************************************************************/
        public static void rbfunserialize(alglib.serializer s,
            rbfmodel model)
        {
            int i0 = 0;
            int i1 = 0;
            int nx = 0;
            int ny = 0;

            
            //
            // Header
            //
            i0 = s.unserialize_int();
            alglib.ap.assert(i0==scodes.getrbfserializationcode(), "RBFUnserialize: stream header corrupted");
            i1 = s.unserialize_int();
            alglib.ap.assert(i1==rbffirstversion, "RBFUnserialize: stream header corrupted");
            
            //
            // Unserialize primary model parameters, initialize model.
            //
            // It is necessary to call RBFCreate() because some internal fields
            // which are NOT unserialized will need initialization.
            //
            nx = s.unserialize_int();
            ny = s.unserialize_int();
            rbfcreate(nx, ny, model);
            model.nc = s.unserialize_int();
            model.nl = s.unserialize_int();
            nearestneighbor.kdtreeunserialize(s, model.tree);
            apserv.unserializerealmatrix(s, ref model.xc);
            apserv.unserializerealmatrix(s, ref model.wr);
            model.rmax = s.unserialize_double();
            apserv.unserializerealmatrix(s, ref model.v);
        }


        /*************************************************************************
        This function changes centers allocation algorithm to one which  allocates
        centers on a regular grid with step equal to H  in  all  dimensions.  This
        function won't have effect until next call to RBFBuildModel().

        Central  point  of  the  grid  is  located  exactly  in  the middle of the 
        bounding rectangle for our dataset. Grid countinues from the center to the
        bounds of the bounding rectangle, and makes one step beyond  bounds  (i.e. 
        leftmost/rightmost/topmost/... lines of the grid are located just  outside
        of the bounding rectangle).

        INPUT PARAMETERS:
            S       -   RBF model, initialized by RBFCreate() call.
            H       -   grid step, H>0

        NOTE: this   function  has   some   serialization-related  subtleties.  We
              recommend you to study serialization examples from ALGLIB  Reference
              Manual if you want to perform serialization of your models.

          -- ALGLIB --
             Copyright 13.12.2011 by Bochkanov Sergey
        *************************************************************************/
        private static void rbfgridregular(rbfmodel s,
            double h)
        {
            alglib.ap.assert(math.isfinite(h) && (double)(h)>(double)(0), "RBFGridRegular: H<0, INF or NAN");
            s.h = h;
            s.gridtype = 1;
        }


        /*************************************************************************
        This function changes centers allocation algorithm to one which  allocates
        centers exactly at the dataset points (one input point = one center). This
        function won't have effect until next call to RBFBuildModel().

        INPUT PARAMETERS:
            S       -   RBF model, initialized by RBFCreate() call

        NOTE: this   function  has   some   serialization-related  subtleties.  We
              recommend you to study serialization examples from ALGLIB  Reference
              Manual if you want to perform serialization of your models.

          -- ALGLIB --
             Copyright 13.12.2011 by Bochkanov Sergey
        *************************************************************************/
        private static void rbfgridpoints(rbfmodel s)
        {
            s.gridtype = 2;
        }


        /*************************************************************************
        This function sets centers, defined by user.

        This function overrides results of the previous calls, i.e. multiple calls
        of this function will result in only the last set being added.

        INPUT PARAMETERS:
            S      -   RBF model, initialized by RBFCreate() call.
          CXY      -   centers, array[N,NX]. Centers must be distinct, 
                       non-distinct centers are not supported.
           NC      -   number of centers

        NOTE: this   function  has   some   serialization-related  subtleties.  We
              recommend you to study serialization examples from ALGLIB  Reference
              Manual if you want to perform serialization of your models.

          -- ALGLIB --
             Copyright 13.12.2011 by Bochkanov Sergey
        *************************************************************************/
        private static void rbfgridspecial(rbfmodel s,
            double[,] cxy,
            int nc)
        {
            int i = 0;
            int j = 0;

            alglib.ap.assert(nc>0, "RBFGridSpecial: N<0");
            alglib.ap.assert(nc<=alglib.ap.rows(cxy), "RBFGridSpecial: Length(CXY)<NC");
            s.gridtype = 3;
            s.nc = nc;
            s.xc = new double[s.nc, mxnx];
            for(i=0; i<=s.nc-1; i++)
            {
                for(j=0; j<=mxnx-1; j++)
                {
                    s.xc[i,j] = 0;
                }
                for(j=0; j<=s.nx-1; j++)
                {
                    s.xc[i,j] = cxy[i,j];
                }
            }
        }


        /*************************************************************************
        This function changes radii calculation algorithm to one which  makes  all
        radii equal to the same fixed value R. This  function  won't  have  effect
        until next call to RBFBuildModel().

        IMPORTANT: you should use this function with caution because too  large  R
        will make model fitting algorithm unstable, while too small  R  will  make
        perfect, but useless fit (it will be non-smooth, with sharp  peaks  around
        dataset points).

        INPUT PARAMETERS:
            S       -   RBF model, initialized by RBFCreate() call
            R       -   radius, R>0

        NOTE: this   function  has   some   serialization-related  subtleties.  We
              recommend you to study serialization examples from ALGLIB  Reference
              Manual if you want to perform serialization of your models.

          -- ALGLIB --
             Copyright 13.12.2011 by Bochkanov Sergey
        *************************************************************************/
        private static void rbfradfixed(rbfmodel s,
            double r)
        {
            alglib.ap.assert(math.isfinite(r) && (double)(r)>(double)(0), "RBFRadFixed: R<=0, infinite or NAN");
            s.fixrad = true;
            s.radvalue = r;
        }


        /*************************************************************************
        This function changes radii  calculation  algorithm  to  one  which  makes
        radius for I-th node equal to R[i]=DistNN[i]*Q, where:
        * R[i] is a radius calculated by the algorithm
        * DistNN[i] is distance from I-th center to its nearest neighbor center
        * Q is a scale parameter, which should be within [0.75,1.50], with
          recommended value equal to 1.0
        * after performing radii calculation, radii are transformed  in  order  to
          avoid situation when single outlier has very large radius and influences
          many points across entire dataset. Transformation has following form:
               new_r[i] = min(r[i],Z*median(r[]))
           where r[i] is I-th radius, median()  is  a median  radius across entire
           dataset, Z is user-specified value which controls amount  of  deviation
           from median radius.

        This function won't have effect until next call to RBFBuildModel().

        The idea behind this algorithm is to choose radii corresponding  to  basis
        functions is such way that I-th radius is approximately equal to  distance
        from I-th center to its nearest neighbor. In this case  interactions  with
        distant points will be insignificant, and we  will  get  well  conditioned
        basis.

        Properties of this basis depend on the value of Q:
        * Q<0.75 will give perfectly conditioned basis,  but  terrible  smoothness
          properties (RBF interpolant will have sharp peaks around function values)
        * Q>1.5 will lead to badly conditioned systems and slow convergence of the
          underlying linear solver (although smoothness will be very good)
        * Q around 1.0 gives good balance between smoothness and condition number


        INPUT PARAMETERS:
            S       -   RBF model, initialized by RBFCreate() call
            Q       -   radius coefficient, Q>0
            Z       -   z-parameter, Z>0
            
        Default value of Q is equal to 1.0
        Default value of Z is equal to 5.0

        NOTE: this   function  has   some   serialization-related  subtleties.  We
              recommend you to study serialization examples from ALGLIB  Reference
              Manual if you want to perform serialization of your models.

          -- ALGLIB --
             Copyright 13.12.2011 by Bochkanov Sergey
        *************************************************************************/
        private static void rbfradnn(rbfmodel s,
            double q,
            double z)
        {
            alglib.ap.assert(math.isfinite(q) && (double)(q)>(double)(0), "RBFRadNN: Q<=0, infinite or NAN");
            alglib.ap.assert(math.isfinite(z) && (double)(z)>(double)(0), "RBFRadNN: Z<=0, infinite or NAN");
            s.fixrad = false;
            s.radvalue = q;
            s.radzvalue = z;
        }


        private static bool buildlinearmodel(double[,] x,
            ref double[,] y,
            int n,
            int ny,
            int modeltype,
            ref double[,] v)
        {
            bool result = new bool();
            double[] tmpy = new double[0];
            double[,] a = new double[0,0];
            double scaling = 0;
            double[] shifting = new double[0];
            double mn = 0;
            double mx = 0;
            double[] c = new double[0];
            lsfit.lsfitreport rep = new lsfit.lsfitreport();
            int i = 0;
            int j = 0;
            int k = 0;
            int info = 0;

            v = new double[0,0];

            alglib.ap.assert(n>=0, "BuildLinearModel: N<0");
            alglib.ap.assert(ny>0, "BuildLinearModel: NY<=0");
            
            //
            // Handle degenerate case (N=0)
            //
            result = true;
            v = new double[ny, mxnx+1];
            if( n==0 )
            {
                for(j=0; j<=mxnx; j++)
                {
                    for(i=0; i<=ny-1; i++)
                    {
                        v[i,j] = 0;
                    }
                }
                return result;
            }
            
            //
            // Allocate temporaries
            //
            tmpy = new double[n];
            
            //
            // General linear model.
            //
            if( modeltype==1 )
            {
                
                //
                // Calculate scaling/shifting, transform variables, prepare LLS problem
                //
                a = new double[n, mxnx+1];
                shifting = new double[mxnx];
                scaling = 0;
                for(i=0; i<=mxnx-1; i++)
                {
                    mn = x[0,i];
                    mx = mn;
                    for(j=1; j<=n-1; j++)
                    {
                        if( (double)(mn)>(double)(x[j,i]) )
                        {
                            mn = x[j,i];
                        }
                        if( (double)(mx)<(double)(x[j,i]) )
                        {
                            mx = x[j,i];
                        }
                    }
                    scaling = Math.Max(scaling, mx-mn);
                    shifting[i] = 0.5*(mx+mn);
                }
                if( (double)(scaling)==(double)(0) )
                {
                    scaling = 1;
                }
                else
                {
                    scaling = 0.5*scaling;
                }
                for(i=0; i<=n-1; i++)
                {
                    for(j=0; j<=mxnx-1; j++)
                    {
                        a[i,j] = (x[i,j]-shifting[j])/scaling;
                    }
                }
                for(i=0; i<=n-1; i++)
                {
                    a[i,mxnx] = 1;
                }
                
                //
                // Solve linear system in transformed variables, make backward 
                //
                for(i=0; i<=ny-1; i++)
                {
                    for(j=0; j<=n-1; j++)
                    {
                        tmpy[j] = y[j,i];
                    }
                    lsfit.lsfitlinear(tmpy, a, n, mxnx+1, ref info, ref c, rep);
                    if( info<=0 )
                    {
                        result = false;
                        return result;
                    }
                    for(j=0; j<=mxnx-1; j++)
                    {
                        v[i,j] = c[j]/scaling;
                    }
                    v[i,mxnx] = c[mxnx];
                    for(j=0; j<=mxnx-1; j++)
                    {
                        v[i,mxnx] = v[i,mxnx]-shifting[j]*v[i,j];
                    }
                    for(j=0; j<=n-1; j++)
                    {
                        for(k=0; k<=mxnx-1; k++)
                        {
                            y[j,i] = y[j,i]-x[j,k]*v[i,k];
                        }
                        y[j,i] = y[j,i]-v[i,mxnx];
                    }
                }
                return result;
            }
            
            //
            // Constant model, very simple
            //
            if( modeltype==2 )
            {
                for(i=0; i<=ny-1; i++)
                {
                    for(j=0; j<=mxnx; j++)
                    {
                        v[i,j] = 0;
                    }
                    for(j=0; j<=n-1; j++)
                    {
                        v[i,mxnx] = v[i,mxnx]+y[j,i];
                    }
                    if( n>0 )
                    {
                        v[i,mxnx] = v[i,mxnx]/n;
                    }
                    for(j=0; j<=n-1; j++)
                    {
                        y[j,i] = y[j,i]-v[i,mxnx];
                    }
                }
                return result;
            }
            
            //
            // Zero model
            //
            alglib.ap.assert(modeltype==3, "BuildLinearModel: unknown model type");
            for(i=0; i<=ny-1; i++)
            {
                for(j=0; j<=mxnx; j++)
                {
                    v[i,j] = 0;
                }
            }
            return result;
        }


        private static void buildrbfmodellsqr(double[,] x,
            ref double[,] y,
            double[,] xc,
            double[] r,
            int n,
            int nc,
            int ny,
            nearestneighbor.kdtree pointstree,
            nearestneighbor.kdtree centerstree,
            double epsort,
            double epserr,
            int maxits,
            ref int gnnz,
            ref int snnz,
            ref double[,] w,
            ref int info,
            ref int iterationscount,
            ref int nmv)
        {
            linlsqr.linlsqrstate state = new linlsqr.linlsqrstate();
            linlsqr.linlsqrreport lsqrrep = new linlsqr.linlsqrreport();
            sparse.sparsematrix spg = new sparse.sparsematrix();
            sparse.sparsematrix sps = new sparse.sparsematrix();
            int[] nearcenterscnt = new int[0];
            int[] nearpointscnt = new int[0];
            int[] skipnearpointscnt = new int[0];
            int[] farpointscnt = new int[0];
            int maxnearcenterscnt = 0;
            int maxnearpointscnt = 0;
            int maxfarpointscnt = 0;
            int sumnearcenterscnt = 0;
            int sumnearpointscnt = 0;
            int sumfarpointscnt = 0;
            double maxrad = 0;
            int[] pointstags = new int[0];
            int[] centerstags = new int[0];
            double[,] nearpoints = new double[0,0];
            double[,] nearcenters = new double[0,0];
            double[,] farpoints = new double[0,0];
            int tmpi = 0;
            int pointscnt = 0;
            int centerscnt = 0;
            double[] xcx = new double[0];
            double[] tmpy = new double[0];
            double[] tc = new double[0];
            double[] g = new double[0];
            double[] c = new double[0];
            int i = 0;
            int j = 0;
            int k = 0;
            int sind = 0;
            double[,] a = new double[0,0];
            double vv = 0;
            double vx = 0;
            double vy = 0;
            double vz = 0;
            double vr = 0;
            double gnorm2 = 0;
            double[] tmp0 = new double[0];
            double[] tmp1 = new double[0];
            double[] tmp2 = new double[0];
            double fx = 0;
            double[,] xx = new double[0,0];
            double[,] cx = new double[0,0];
            double mrad = 0;
            int i_ = 0;

            gnnz = 0;
            snnz = 0;
            w = new double[0,0];
            info = 0;
            iterationscount = 0;
            nmv = 0;

            
            //
            // Handle special cases: NC=0
            //
            if( nc==0 )
            {
                info = 1;
                iterationscount = 0;
                nmv = 0;
                return;
            }
            
            //
            // Prepare for general case, NC>0
            //
            xcx = new double[mxnx];
            pointstags = new int[n];
            centerstags = new int[nc];
            info = -1;
            iterationscount = 0;
            nmv = 0;
            
            //
            // This block prepares quantities used to compute approximate cardinal basis functions (ACBFs):
            // * NearCentersCnt[]   -   array[NC], whose elements store number of near centers used to build ACBF
            // * NearPointsCnt[]    -   array[NC], number of near points used to build ACBF
            // * FarPointsCnt[]     -   array[NC], number of far points (ones where ACBF is nonzero)
            // * MaxNearCentersCnt  -   max(NearCentersCnt)
            // * MaxNearPointsCnt   -   max(NearPointsCnt)
            // * SumNearCentersCnt  -   sum(NearCentersCnt)
            // * SumNearPointsCnt   -   sum(NearPointsCnt)
            // * SumFarPointsCnt    -   sum(FarPointsCnt)
            //
            nearcenterscnt = new int[nc];
            nearpointscnt = new int[nc];
            skipnearpointscnt = new int[nc];
            farpointscnt = new int[nc];
            maxnearcenterscnt = 0;
            maxnearpointscnt = 0;
            maxfarpointscnt = 0;
            sumnearcenterscnt = 0;
            sumnearpointscnt = 0;
            sumfarpointscnt = 0;
            for(i=0; i<=nc-1; i++)
            {
                for(j=0; j<=mxnx-1; j++)
                {
                    xcx[j] = xc[i,j];
                }
                
                //
                // Determine number of near centers and maximum radius of near centers
                //
                nearcenterscnt[i] = nearestneighbor.kdtreequeryrnn(centerstree, xcx, r[i]*rbfnearradius, true);
                nearestneighbor.kdtreequeryresultstags(centerstree, ref centerstags);
                maxrad = 0;
                for(j=0; j<=nearcenterscnt[i]-1; j++)
                {
                    maxrad = Math.Max(maxrad, Math.Abs(r[centerstags[j]]));
                }
                
                //
                // Determine number of near points (ones which used to build ACBF)
                // and skipped points (the most near points which are NOT used to build ACBF
                // and are NOT included in the near points count
                //
                skipnearpointscnt[i] = nearestneighbor.kdtreequeryrnn(pointstree, xcx, 0.1*r[i], true);
                nearpointscnt[i] = nearestneighbor.kdtreequeryrnn(pointstree, xcx, (r[i]+maxrad)*rbfnearradius, true)-skipnearpointscnt[i];
                alglib.ap.assert(nearpointscnt[i]>=0, "BuildRBFModelLSQR: internal error");
                
                //
                // Determine number of far points
                //
                farpointscnt[i] = nearestneighbor.kdtreequeryrnn(pointstree, xcx, Math.Max(r[i]*rbfnearradius+maxrad*rbffarradius, r[i]*rbffarradius), true);
                
                //
                // calculate sum and max, make some basic checks
                //
                alglib.ap.assert(nearcenterscnt[i]>0, "BuildRBFModelLSQR: internal error");
                maxnearcenterscnt = Math.Max(maxnearcenterscnt, nearcenterscnt[i]);
                maxnearpointscnt = Math.Max(maxnearpointscnt, nearpointscnt[i]);
                maxfarpointscnt = Math.Max(maxfarpointscnt, farpointscnt[i]);
                sumnearcenterscnt = sumnearcenterscnt+nearcenterscnt[i];
                sumnearpointscnt = sumnearpointscnt+nearpointscnt[i];
                sumfarpointscnt = sumfarpointscnt+farpointscnt[i];
            }
            snnz = sumnearcenterscnt;
            gnnz = sumfarpointscnt;
            alglib.ap.assert(maxnearcenterscnt>0, "BuildRBFModelLSQR: internal error");
            
            //
            // Allocate temporaries.
            //
            // NOTE: we want to avoid allocation of zero-size arrays, so we
            //       use max(desired_size,1) instead of desired_size when performing
            //       memory allocation.
            //
            a = new double[maxnearpointscnt+maxnearcenterscnt, maxnearcenterscnt];
            tmpy = new double[maxnearpointscnt+maxnearcenterscnt];
            g = new double[maxnearcenterscnt];
            c = new double[maxnearcenterscnt];
            nearcenters = new double[maxnearcenterscnt, mxnx];
            nearpoints = new double[Math.Max(maxnearpointscnt, 1), mxnx];
            farpoints = new double[Math.Max(maxfarpointscnt, 1), mxnx];
            
            //
            // fill matrix SpG
            //
            sparse.sparsecreate(n, nc, gnnz, spg);
            sparse.sparsecreate(nc, nc, snnz, sps);
            for(i=0; i<=nc-1; i++)
            {
                centerscnt = nearcenterscnt[i];
                
                //
                // main center
                //
                for(j=0; j<=mxnx-1; j++)
                {
                    xcx[j] = xc[i,j];
                }
                
                //
                // center's tree
                //
                tmpi = nearestneighbor.kdtreequeryknn(centerstree, xcx, centerscnt, true);
                alglib.ap.assert(tmpi==centerscnt, "BuildRBFModelLSQR: internal error");
                nearestneighbor.kdtreequeryresultsx(centerstree, ref cx);
                nearestneighbor.kdtreequeryresultstags(centerstree, ref centerstags);
                
                //
                // point's tree
                //
                mrad = 0;
                for(j=0; j<=centerscnt-1; j++)
                {
                    mrad = Math.Max(mrad, r[centerstags[j]]);
                }
                
                //
                // we need to be sure that 'CTree' contains
                // at least one side center
                //
                sparse.sparseset(sps, i, i, 1);
                c[0] = 1.0;
                for(j=1; j<=centerscnt-1; j++)
                {
                    c[j] = 0.0;
                }
                if( centerscnt>1 && nearpointscnt[i]>0 )
                {
                    
                    //
                    // first KDTree request for points
                    //
                    pointscnt = nearpointscnt[i];
                    tmpi = nearestneighbor.kdtreequeryknn(pointstree, xcx, skipnearpointscnt[i]+nearpointscnt[i], true);
                    alglib.ap.assert(tmpi==skipnearpointscnt[i]+nearpointscnt[i], "BuildRBFModelLSQR: internal error");
                    nearestneighbor.kdtreequeryresultsx(pointstree, ref xx);
                    sind = skipnearpointscnt[i];
                    for(j=0; j<=pointscnt-1; j++)
                    {
                        vx = xx[sind+j,0];
                        vy = xx[sind+j,1];
                        vz = xx[sind+j,2];
                        for(k=0; k<=centerscnt-1; k++)
                        {
                            vr = 0.0;
                            vv = vx-cx[k,0];
                            vr = vr+vv*vv;
                            vv = vy-cx[k,1];
                            vr = vr+vv*vv;
                            vv = vz-cx[k,2];
                            vr = vr+vv*vv;
                            vv = r[centerstags[k]];
                            a[j,k] = Math.Exp(-(vr/(vv*vv)));
                        }
                    }
                    for(j=0; j<=centerscnt-1; j++)
                    {
                        g[j] = Math.Exp(-((math.sqr(xcx[0]-cx[j,0])+math.sqr(xcx[1]-cx[j,1])+math.sqr(xcx[2]-cx[j,2]))/math.sqr(r[centerstags[j]])));
                    }
                    
                    //
                    // calculate the problem
                    //
                    gnorm2 = 0.0;
                    for(i_=0; i_<=centerscnt-1;i_++)
                    {
                        gnorm2 += g[i_]*g[i_];
                    }
                    for(j=0; j<=pointscnt-1; j++)
                    {
                        vv = 0.0;
                        for(i_=0; i_<=centerscnt-1;i_++)
                        {
                            vv += a[j,i_]*g[i_];
                        }
                        vv = vv/gnorm2;
                        tmpy[j] = -vv;
                        for(i_=0; i_<=centerscnt-1;i_++)
                        {
                            a[j,i_] = a[j,i_] - vv*g[i_];
                        }
                    }
                    for(j=pointscnt; j<=pointscnt+centerscnt-1; j++)
                    {
                        for(k=0; k<=centerscnt-1; k++)
                        {
                            a[j,k] = 0.0;
                        }
                        a[j,j-pointscnt] = 1.0E-6;
                        tmpy[j] = 0.0;
                    }
                    fbls.fblssolvels(ref a, ref tmpy, pointscnt+centerscnt, centerscnt, ref tmp0, ref tmp1, ref tmp2);
                    for(i_=0; i_<=centerscnt-1;i_++)
                    {
                        c[i_] = tmpy[i_];
                    }
                    vv = 0.0;
                    for(i_=0; i_<=centerscnt-1;i_++)
                    {
                        vv += g[i_]*c[i_];
                    }
                    vv = vv/gnorm2;
                    for(i_=0; i_<=centerscnt-1;i_++)
                    {
                        c[i_] = c[i_] - vv*g[i_];
                    }
                    vv = 1/gnorm2;
                    for(i_=0; i_<=centerscnt-1;i_++)
                    {
                        c[i_] = c[i_] + vv*g[i_];
                    }
                    for(j=0; j<=centerscnt-1; j++)
                    {
                        sparse.sparseset(sps, i, centerstags[j], c[j]);
                    }
                }
                
                //
                // second KDTree request for points
                //
                pointscnt = farpointscnt[i];
                tmpi = nearestneighbor.kdtreequeryknn(pointstree, xcx, pointscnt, true);
                alglib.ap.assert(tmpi==pointscnt, "BuildRBFModelLSQR: internal error");
                nearestneighbor.kdtreequeryresultsx(pointstree, ref xx);
                nearestneighbor.kdtreequeryresultstags(pointstree, ref pointstags);
                
                //
                //fill SpG matrix
                //
                for(j=0; j<=pointscnt-1; j++)
                {
                    fx = 0;
                    vx = xx[j,0];
                    vy = xx[j,1];
                    vz = xx[j,2];
                    for(k=0; k<=centerscnt-1; k++)
                    {
                        vr = 0.0;
                        vv = vx-cx[k,0];
                        vr = vr+vv*vv;
                        vv = vy-cx[k,1];
                        vr = vr+vv*vv;
                        vv = vz-cx[k,2];
                        vr = vr+vv*vv;
                        vv = r[centerstags[k]];
                        vv = vv*vv;
                        fx = fx+c[k]*Math.Exp(-(vr/vv));
                    }
                    sparse.sparseset(spg, pointstags[j], i, fx);
                }
            }
            sparse.sparseconverttocrs(spg);
            sparse.sparseconverttocrs(sps);
            
            //
            // solve by LSQR method
            //
            tmpy = new double[n];
            tc = new double[nc];
            w = new double[nc, ny];
            linlsqr.linlsqrcreate(n, nc, state);
            linlsqr.linlsqrsetcond(state, epsort, epserr, maxits);
            for(i=0; i<=ny-1; i++)
            {
                for(j=0; j<=n-1; j++)
                {
                    tmpy[j] = y[j,i];
                }
                linlsqr.linlsqrsolvesparse(state, spg, tmpy);
                linlsqr.linlsqrresults(state, ref c, lsqrrep);
                if( lsqrrep.terminationtype<=0 )
                {
                    info = -4;
                    return;
                }
                sparse.sparsemtv(sps, c, ref tc);
                for(j=0; j<=nc-1; j++)
                {
                    w[j,i] = tc[j];
                }
                iterationscount = iterationscount+lsqrrep.iterationscount;
                nmv = nmv+lsqrrep.nmv;
            }
            info = 1;
        }


        private static void buildrbfmlayersmodellsqr(double[,] x,
            ref double[,] y,
            ref double[,] xc,
            double rval,
            ref double[] r,
            int n,
            ref int nc,
            int ny,
            int nlayers,
            nearestneighbor.kdtree centerstree,
            double epsort,
            double epserr,
            int maxits,
            double lambdav,
            ref int annz,
            ref double[,] w,
            ref int info,
            ref int iterationscount,
            ref int nmv)
        {
            linlsqr.linlsqrstate state = new linlsqr.linlsqrstate();
            linlsqr.linlsqrreport lsqrrep = new linlsqr.linlsqrreport();
            sparse.sparsematrix spa = new sparse.sparsematrix();
            double anorm = 0;
            double[] omega = new double[0];
            double[] xx = new double[0];
            double[] tmpy = new double[0];
            double[,] cx = new double[0,0];
            double yval = 0;
            int nec = 0;
            int[] centerstags = new int[0];
            int layer = 0;
            int i = 0;
            int j = 0;
            int k = 0;
            double v = 0;
            double rmaxbefore = 0;
            double rmaxafter = 0;

            xc = new double[0,0];
            r = new double[0];
            nc = 0;
            annz = 0;
            w = new double[0,0];
            info = 0;
            iterationscount = 0;
            nmv = 0;

            alglib.ap.assert(nlayers>=0, "BuildRBFMLayersModelLSQR: invalid argument(NLayers<0)");
            alglib.ap.assert(n>=0, "BuildRBFMLayersModelLSQR: invalid argument(N<0)");
            alglib.ap.assert(mxnx>0 && mxnx<=3, "BuildRBFMLayersModelLSQR: internal error(invalid global const MxNX: either MxNX<=0 or MxNX>3)");
            annz = 0;
            if( n==0 || nlayers==0 )
            {
                info = 1;
                iterationscount = 0;
                nmv = 0;
                return;
            }
            nc = n*nlayers;
            xx = new double[mxnx];
            centerstags = new int[n];
            xc = new double[nc, mxnx];
            r = new double[nc];
            for(i=0; i<=nc-1; i++)
            {
                for(j=0; j<=mxnx-1; j++)
                {
                    xc[i,j] = x[i%n,j];
                }
            }
            for(i=0; i<=nc-1; i++)
            {
                r[i] = rval/Math.Pow(2, i/n);
            }
            for(i=0; i<=n-1; i++)
            {
                centerstags[i] = i;
            }
            nearestneighbor.kdtreebuildtagged(xc, centerstags, n, mxnx, 0, 2, centerstree);
            omega = new double[n];
            tmpy = new double[n];
            w = new double[nc, ny];
            info = -1;
            iterationscount = 0;
            nmv = 0;
            linlsqr.linlsqrcreate(n, n, state);
            linlsqr.linlsqrsetcond(state, epsort, epserr, maxits);
            linlsqr.linlsqrsetlambdai(state, 1.0E-6);
            
            //
            // calculate number of non-zero elements for sparse matrix
            //
            for(i=0; i<=n-1; i++)
            {
                for(j=0; j<=mxnx-1; j++)
                {
                    xx[j] = x[i,j];
                }
                annz = annz+nearestneighbor.kdtreequeryrnn(centerstree, xx, r[0]*rbfmlradius, true);
            }
            for(layer=0; layer<=nlayers-1; layer++)
            {
                
                //
                // Fill sparse matrix, calculate norm(A)
                //
                anorm = 0.0;
                sparse.sparsecreate(n, n, annz, spa);
                for(i=0; i<=n-1; i++)
                {
                    for(j=0; j<=mxnx-1; j++)
                    {
                        xx[j] = x[i,j];
                    }
                    nec = nearestneighbor.kdtreequeryrnn(centerstree, xx, r[layer*n]*rbfmlradius, true);
                    nearestneighbor.kdtreequeryresultsx(centerstree, ref cx);
                    nearestneighbor.kdtreequeryresultstags(centerstree, ref centerstags);
                    for(j=0; j<=nec-1; j++)
                    {
                        v = Math.Exp(-((math.sqr(xx[0]-cx[j,0])+math.sqr(xx[1]-cx[j,1])+math.sqr(xx[2]-cx[j,2]))/math.sqr(r[layer*n+centerstags[j]])));
                        sparse.sparseset(spa, i, centerstags[j], v);
                        anorm = anorm+math.sqr(v);
                    }
                }
                anorm = Math.Sqrt(anorm);
                sparse.sparseconverttocrs(spa);
                
                //
                // Calculate maximum residual before adding new layer.
                // This value is not used by algorithm, the only purpose is to make debugging easier.
                //
                rmaxbefore = 0.0;
                for(j=0; j<=n-1; j++)
                {
                    for(i=0; i<=ny-1; i++)
                    {
                        rmaxbefore = Math.Max(rmaxbefore, Math.Abs(y[j,i]));
                    }
                }
                
                //
                // Process NY dimensions of the target function
                //
                for(i=0; i<=ny-1; i++)
                {
                    for(j=0; j<=n-1; j++)
                    {
                        tmpy[j] = y[j,i];
                    }
                    
                    //
                    // calculate Omega for current layer
                    //
                    linlsqr.linlsqrsetlambdai(state, lambdav*anorm/n);
                    linlsqr.linlsqrsolvesparse(state, spa, tmpy);
                    linlsqr.linlsqrresults(state, ref omega, lsqrrep);
                    if( lsqrrep.terminationtype<=0 )
                    {
                        info = -4;
                        return;
                    }
                    
                    //
                    // calculate error for current layer
                    //
                    for(j=0; j<=n-1; j++)
                    {
                        yval = 0;
                        for(k=0; k<=mxnx-1; k++)
                        {
                            xx[k] = x[j,k];
                        }
                        nec = nearestneighbor.kdtreequeryrnn(centerstree, xx, r[layer*n]*rbffarradius, true);
                        nearestneighbor.kdtreequeryresultsx(centerstree, ref cx);
                        nearestneighbor.kdtreequeryresultstags(centerstree, ref centerstags);
                        for(k=0; k<=nec-1; k++)
                        {
                            yval = yval+omega[centerstags[k]]*Math.Exp(-((math.sqr(xx[0]-cx[k,0])+math.sqr(xx[1]-cx[k,1])+math.sqr(xx[2]-cx[k,2]))/math.sqr(r[layer*n+centerstags[k]])));
                        }
                        y[j,i] = y[j,i]-yval;
                    }
                    
                    //
                    // write Omega in out parameter W
                    //
                    for(j=0; j<=n-1; j++)
                    {
                        w[layer*n+j,i] = omega[j];
                    }
                    iterationscount = iterationscount+lsqrrep.iterationscount;
                    nmv = nmv+lsqrrep.nmv;
                }
                
                //
                // Calculate maximum residual before adding new layer.
                // This value is not used by algorithm, the only purpose is to make debugging easier.
                //
                rmaxafter = 0.0;
                for(j=0; j<=n-1; j++)
                {
                    for(i=0; i<=ny-1; i++)
                    {
                        rmaxafter = Math.Max(rmaxafter, Math.Abs(y[j,i]));
                    }
                }
            }
            info = 1;
        }


    }
    public class spline2d
    {
        /*************************************************************************
        2-dimensional spline inteprolant
        *************************************************************************/
        public class spline2dinterpolant
        {
            public int k;
            public double[] c;
            public spline2dinterpolant()
            {
                c = new double[0];
            }
        };




        /*************************************************************************
        This subroutine builds bilinear spline coefficients table.

        Input parameters:
            X   -   spline abscissas, array[0..N-1]
            Y   -   spline ordinates, array[0..M-1]
            F   -   function values, array[0..M-1,0..N-1]
            M,N -   grid size, M>=2, N>=2

        Output parameters:
            C   -   spline interpolant

          -- ALGLIB PROJECT --
             Copyright 05.07.2007 by Bochkanov Sergey
        *************************************************************************/
        public static void spline2dbuildbilinear(double[] x,
            double[] y,
            double[,] f,
            int m,
            int n,
            spline2dinterpolant c)
        {
            int i = 0;
            int j = 0;
            int k = 0;
            int tblsize = 0;
            int shift = 0;
            double t = 0;
            double[,] dx = new double[0,0];
            double[,] dy = new double[0,0];
            double[,] dxy = new double[0,0];

            x = (double[])x.Clone();
            y = (double[])y.Clone();
            f = (double[,])f.Clone();

            alglib.ap.assert(n>=2 && m>=2, "Spline2DBuildBilinear: N<2 or M<2!");
            
            //
            // Sort points
            //
            for(j=0; j<=n-1; j++)
            {
                k = j;
                for(i=j+1; i<=n-1; i++)
                {
                    if( (double)(x[i])<(double)(x[k]) )
                    {
                        k = i;
                    }
                }
                if( k!=j )
                {
                    for(i=0; i<=m-1; i++)
                    {
                        t = f[i,j];
                        f[i,j] = f[i,k];
                        f[i,k] = t;
                    }
                    t = x[j];
                    x[j] = x[k];
                    x[k] = t;
                }
            }
            for(i=0; i<=m-1; i++)
            {
                k = i;
                for(j=i+1; j<=m-1; j++)
                {
                    if( (double)(y[j])<(double)(y[k]) )
                    {
                        k = j;
                    }
                }
                if( k!=i )
                {
                    for(j=0; j<=n-1; j++)
                    {
                        t = f[i,j];
                        f[i,j] = f[k,j];
                        f[k,j] = t;
                    }
                    t = y[i];
                    y[i] = y[k];
                    y[k] = t;
                }
            }
            
            //
            // Fill C:
            //  C[0]            -   length(C)
            //  C[1]            -   type(C):
            //                      -1 = bilinear interpolant
            //                      -3 = general cubic spline
            //                           (see BuildBicubicSpline)
            //  C[2]:
            //      N (x count)
            //  C[3]:
            //      M (y count)
            //  C[4]...C[4+N-1]:
            //      x[i], i = 0...N-1
            //  C[4+N]...C[4+N+M-1]:
            //      y[i], i = 0...M-1
            //  C[4+N+M]...C[4+N+M+(N*M-1)]:
            //      f(i,j) table. f(0,0), f(0, 1), f(0,2) and so on...
            //
            c.k = 1;
            tblsize = 4+n+m+n*m;
            c.c = new double[tblsize-1+1];
            c.c[0] = tblsize;
            c.c[1] = -1;
            c.c[2] = n;
            c.c[3] = m;
            for(i=0; i<=n-1; i++)
            {
                c.c[4+i] = x[i];
            }
            for(i=0; i<=m-1; i++)
            {
                c.c[4+n+i] = y[i];
            }
            for(i=0; i<=m-1; i++)
            {
                for(j=0; j<=n-1; j++)
                {
                    shift = i*n+j;
                    c.c[4+n+m+shift] = f[i,j];
                }
            }
        }


        /*************************************************************************
        This subroutine builds bicubic spline coefficients table.

        Input parameters:
            X   -   spline abscissas, array[0..N-1]
            Y   -   spline ordinates, array[0..M-1]
            F   -   function values, array[0..M-1,0..N-1]
            M,N -   grid size, M>=2, N>=2

        Output parameters:
            C   -   spline interpolant

          -- ALGLIB PROJECT --
             Copyright 05.07.2007 by Bochkanov Sergey
        *************************************************************************/
        public static void spline2dbuildbicubic(double[] x,
            double[] y,
            double[,] f,
            int m,
            int n,
            spline2dinterpolant c)
        {
            int i = 0;
            int j = 0;
            int k = 0;
            int tblsize = 0;
            int shift = 0;
            double t = 0;
            double[,] dx = new double[0,0];
            double[,] dy = new double[0,0];
            double[,] dxy = new double[0,0];

            x = (double[])x.Clone();
            y = (double[])y.Clone();
            f = (double[,])f.Clone();

            alglib.ap.assert(n>=2 && m>=2, "BuildBicubicSpline: N<2 or M<2!");
            
            //
            // Sort points
            //
            for(j=0; j<=n-1; j++)
            {
                k = j;
                for(i=j+1; i<=n-1; i++)
                {
                    if( (double)(x[i])<(double)(x[k]) )
                    {
                        k = i;
                    }
                }
                if( k!=j )
                {
                    for(i=0; i<=m-1; i++)
                    {
                        t = f[i,j];
                        f[i,j] = f[i,k];
                        f[i,k] = t;
                    }
                    t = x[j];
                    x[j] = x[k];
                    x[k] = t;
                }
            }
            for(i=0; i<=m-1; i++)
            {
                k = i;
                for(j=i+1; j<=m-1; j++)
                {
                    if( (double)(y[j])<(double)(y[k]) )
                    {
                        k = j;
                    }
                }
                if( k!=i )
                {
                    for(j=0; j<=n-1; j++)
                    {
                        t = f[i,j];
                        f[i,j] = f[k,j];
                        f[k,j] = t;
                    }
                    t = y[i];
                    y[i] = y[k];
                    y[k] = t;
                }
            }
            
            //
            // Fill C:
            //  C[0]            -   length(C)
            //  C[1]            -   type(C):
            //                      -1 = bilinear interpolant
            //                           (see BuildBilinearInterpolant)
            //                      -3 = general cubic spline
            //  C[2]:
            //      N (x count)
            //  C[3]:
            //      M (y count)
            //  C[4]...C[4+N-1]:
            //      x[i], i = 0...N-1
            //  C[4+N]...C[4+N+M-1]:
            //      y[i], i = 0...M-1
            //  C[4+N+M]...C[4+N+M+(N*M-1)]:
            //      f(i,j) table. f(0,0), f(0, 1), f(0,2) and so on...
            //  C[4+N+M+N*M]...C[4+N+M+(2*N*M-1)]:
            //      df(i,j)/dx table.
            //  C[4+N+M+2*N*M]...C[4+N+M+(3*N*M-1)]:
            //      df(i,j)/dy table.
            //  C[4+N+M+3*N*M]...C[4+N+M+(4*N*M-1)]:
            //      d2f(i,j)/dxdy table.
            //
            c.k = 3;
            tblsize = 4+n+m+4*n*m;
            c.c = new double[tblsize-1+1];
            c.c[0] = tblsize;
            c.c[1] = -3;
            c.c[2] = n;
            c.c[3] = m;
            for(i=0; i<=n-1; i++)
            {
                c.c[4+i] = x[i];
            }
            for(i=0; i<=m-1; i++)
            {
                c.c[4+n+i] = y[i];
            }
            bicubiccalcderivatives(f, x, y, m, n, ref dx, ref dy, ref dxy);
            for(i=0; i<=m-1; i++)
            {
                for(j=0; j<=n-1; j++)
                {
                    shift = i*n+j;
                    c.c[4+n+m+shift] = f[i,j];
                    c.c[4+n+m+n*m+shift] = dx[i,j];
                    c.c[4+n+m+2*n*m+shift] = dy[i,j];
                    c.c[4+n+m+3*n*m+shift] = dxy[i,j];
                }
            }
        }


        /*************************************************************************
        This subroutine calculates the value of the bilinear or bicubic spline  at
        the given point X.

        Input parameters:
            C   -   coefficients table.
                    Built by BuildBilinearSpline or BuildBicubicSpline.
            X, Y-   point

        Result:
            S(x,y)

          -- ALGLIB PROJECT --
             Copyright 05.07.2007 by Bochkanov Sergey
        *************************************************************************/
        public static double spline2dcalc(spline2dinterpolant c,
            double x,
            double y)
        {
            double result = 0;
            double v = 0;
            double vx = 0;
            double vy = 0;
            double vxy = 0;

            spline2ddiff(c, x, y, ref v, ref vx, ref vy, ref vxy);
            result = v;
            return result;
        }


        /*************************************************************************
        This subroutine calculates the value of the bilinear or bicubic spline  at
        the given point X and its derivatives.

        Input parameters:
            C   -   spline interpolant.
            X, Y-   point

        Output parameters:
            F   -   S(x,y)
            FX  -   dS(x,y)/dX
            FY  -   dS(x,y)/dY
            FXY -   d2S(x,y)/dXdY

          -- ALGLIB PROJECT --
             Copyright 05.07.2007 by Bochkanov Sergey
        *************************************************************************/
        public static void spline2ddiff(spline2dinterpolant c,
            double x,
            double y,
            ref double f,
            ref double fx,
            ref double fy,
            ref double fxy)
        {
            int n = 0;
            int m = 0;
            double t = 0;
            double dt = 0;
            double u = 0;
            double du = 0;
            int ix = 0;
            int iy = 0;
            int l = 0;
            int r = 0;
            int h = 0;
            int shift1 = 0;
            int s1 = 0;
            int s2 = 0;
            int s3 = 0;
            int s4 = 0;
            int sf = 0;
            int sfx = 0;
            int sfy = 0;
            int sfxy = 0;
            double y1 = 0;
            double y2 = 0;
            double y3 = 0;
            double y4 = 0;
            double v = 0;
            double t0 = 0;
            double t1 = 0;
            double t2 = 0;
            double t3 = 0;
            double u0 = 0;
            double u1 = 0;
            double u2 = 0;
            double u3 = 0;

            f = 0;
            fx = 0;
            fy = 0;
            fxy = 0;

            alglib.ap.assert((int)Math.Round(c.c[1])==-1 || (int)Math.Round(c.c[1])==-3, "Spline2DDiff: incorrect C!");
            n = (int)Math.Round(c.c[2]);
            m = (int)Math.Round(c.c[3]);
            
            //
            // Binary search in the [ x[0], ..., x[n-2] ] (x[n-1] is not included)
            //
            l = 4;
            r = 4+n-2+1;
            while( l!=r-1 )
            {
                h = (l+r)/2;
                if( (double)(c.c[h])>=(double)(x) )
                {
                    r = h;
                }
                else
                {
                    l = h;
                }
            }
            t = (x-c.c[l])/(c.c[l+1]-c.c[l]);
            dt = 1.0/(c.c[l+1]-c.c[l]);
            ix = l-4;
            
            //
            // Binary search in the [ y[0], ..., y[m-2] ] (y[m-1] is not included)
            //
            l = 4+n;
            r = 4+n+(m-2)+1;
            while( l!=r-1 )
            {
                h = (l+r)/2;
                if( (double)(c.c[h])>=(double)(y) )
                {
                    r = h;
                }
                else
                {
                    l = h;
                }
            }
            u = (y-c.c[l])/(c.c[l+1]-c.c[l]);
            du = 1.0/(c.c[l+1]-c.c[l]);
            iy = l-(4+n);
            
            //
            // Prepare F, dF/dX, dF/dY, d2F/dXdY
            //
            f = 0;
            fx = 0;
            fy = 0;
            fxy = 0;
            
            //
            // Bilinear interpolation
            //
            if( (int)Math.Round(c.c[1])==-1 )
            {
                shift1 = 4+n+m;
                y1 = c.c[shift1+n*iy+ix];
                y2 = c.c[shift1+n*iy+(ix+1)];
                y3 = c.c[shift1+n*(iy+1)+(ix+1)];
                y4 = c.c[shift1+n*(iy+1)+ix];
                f = (1-t)*(1-u)*y1+t*(1-u)*y2+t*u*y3+(1-t)*u*y4;
                fx = (-((1-u)*y1)+(1-u)*y2+u*y3-u*y4)*dt;
                fy = (-((1-t)*y1)-t*y2+t*y3+(1-t)*y4)*du;
                fxy = (y1-y2+y3-y4)*du*dt;
                return;
            }
            
            //
            // Bicubic interpolation
            //
            if( (int)Math.Round(c.c[1])==-3 )
            {
                
                //
                // Prepare info
                //
                t0 = 1;
                t1 = t;
                t2 = math.sqr(t);
                t3 = t*t2;
                u0 = 1;
                u1 = u;
                u2 = math.sqr(u);
                u3 = u*u2;
                sf = 4+n+m;
                sfx = 4+n+m+n*m;
                sfy = 4+n+m+2*n*m;
                sfxy = 4+n+m+3*n*m;
                s1 = n*iy+ix;
                s2 = n*iy+(ix+1);
                s3 = n*(iy+1)+(ix+1);
                s4 = n*(iy+1)+ix;
                
                //
                // Calculate
                //
                v = 1*c.c[sf+s1];
                f = f+v*t0*u0;
                v = 1*c.c[sfy+s1]/du;
                f = f+v*t0*u1;
                fy = fy+1*v*t0*u0*du;
                v = -(3*c.c[sf+s1])+3*c.c[sf+s4]-2*c.c[sfy+s1]/du-1*c.c[sfy+s4]/du;
                f = f+v*t0*u2;
                fy = fy+2*v*t0*u1*du;
                v = 2*c.c[sf+s1]-2*c.c[sf+s4]+1*c.c[sfy+s1]/du+1*c.c[sfy+s4]/du;
                f = f+v*t0*u3;
                fy = fy+3*v*t0*u2*du;
                v = 1*c.c[sfx+s1]/dt;
                f = f+v*t1*u0;
                fx = fx+1*v*t0*u0*dt;
                v = 1*c.c[sfxy+s1]/(dt*du);
                f = f+v*t1*u1;
                fx = fx+1*v*t0*u1*dt;
                fy = fy+1*v*t1*u0*du;
                fxy = fxy+1*v*t0*u0*dt*du;
                v = -(3*c.c[sfx+s1]/dt)+3*c.c[sfx+s4]/dt-2*c.c[sfxy+s1]/(dt*du)-1*c.c[sfxy+s4]/(dt*du);
                f = f+v*t1*u2;
                fx = fx+1*v*t0*u2*dt;
                fy = fy+2*v*t1*u1*du;
                fxy = fxy+2*v*t0*u1*dt*du;
                v = 2*c.c[sfx+s1]/dt-2*c.c[sfx+s4]/dt+1*c.c[sfxy+s1]/(dt*du)+1*c.c[sfxy+s4]/(dt*du);
                f = f+v*t1*u3;
                fx = fx+1*v*t0*u3*dt;
                fy = fy+3*v*t1*u2*du;
                fxy = fxy+3*v*t0*u2*dt*du;
                v = -(3*c.c[sf+s1])+3*c.c[sf+s2]-2*c.c[sfx+s1]/dt-1*c.c[sfx+s2]/dt;
                f = f+v*t2*u0;
                fx = fx+2*v*t1*u0*dt;
                v = -(3*c.c[sfy+s1]/du)+3*c.c[sfy+s2]/du-2*c.c[sfxy+s1]/(dt*du)-1*c.c[sfxy+s2]/(dt*du);
                f = f+v*t2*u1;
                fx = fx+2*v*t1*u1*dt;
                fy = fy+1*v*t2*u0*du;
                fxy = fxy+2*v*t1*u0*dt*du;
                v = 9*c.c[sf+s1]-9*c.c[sf+s2]+9*c.c[sf+s3]-9*c.c[sf+s4]+6*c.c[sfx+s1]/dt+3*c.c[sfx+s2]/dt-3*c.c[sfx+s3]/dt-6*c.c[sfx+s4]/dt+6*c.c[sfy+s1]/du-6*c.c[sfy+s2]/du-3*c.c[sfy+s3]/du+3*c.c[sfy+s4]/du+4*c.c[sfxy+s1]/(dt*du)+2*c.c[sfxy+s2]/(dt*du)+1*c.c[sfxy+s3]/(dt*du)+2*c.c[sfxy+s4]/(dt*du);
                f = f+v*t2*u2;
                fx = fx+2*v*t1*u2*dt;
                fy = fy+2*v*t2*u1*du;
                fxy = fxy+4*v*t1*u1*dt*du;
                v = -(6*c.c[sf+s1])+6*c.c[sf+s2]-6*c.c[sf+s3]+6*c.c[sf+s4]-4*c.c[sfx+s1]/dt-2*c.c[sfx+s2]/dt+2*c.c[sfx+s3]/dt+4*c.c[sfx+s4]/dt-3*c.c[sfy+s1]/du+3*c.c[sfy+s2]/du+3*c.c[sfy+s3]/du-3*c.c[sfy+s4]/du-2*c.c[sfxy+s1]/(dt*du)-1*c.c[sfxy+s2]/(dt*du)-1*c.c[sfxy+s3]/(dt*du)-2*c.c[sfxy+s4]/(dt*du);
                f = f+v*t2*u3;
                fx = fx+2*v*t1*u3*dt;
                fy = fy+3*v*t2*u2*du;
                fxy = fxy+6*v*t1*u2*dt*du;
                v = 2*c.c[sf+s1]-2*c.c[sf+s2]+1*c.c[sfx+s1]/dt+1*c.c[sfx+s2]/dt;
                f = f+v*t3*u0;
                fx = fx+3*v*t2*u0*dt;
                v = 2*c.c[sfy+s1]/du-2*c.c[sfy+s2]/du+1*c.c[sfxy+s1]/(dt*du)+1*c.c[sfxy+s2]/(dt*du);
                f = f+v*t3*u1;
                fx = fx+3*v*t2*u1*dt;
                fy = fy+1*v*t3*u0*du;
                fxy = fxy+3*v*t2*u0*dt*du;
                v = -(6*c.c[sf+s1])+6*c.c[sf+s2]-6*c.c[sf+s3]+6*c.c[sf+s4]-3*c.c[sfx+s1]/dt-3*c.c[sfx+s2]/dt+3*c.c[sfx+s3]/dt+3*c.c[sfx+s4]/dt-4*c.c[sfy+s1]/du+4*c.c[sfy+s2]/du+2*c.c[sfy+s3]/du-2*c.c[sfy+s4]/du-2*c.c[sfxy+s1]/(dt*du)-2*c.c[sfxy+s2]/(dt*du)-1*c.c[sfxy+s3]/(dt*du)-1*c.c[sfxy+s4]/(dt*du);
                f = f+v*t3*u2;
                fx = fx+3*v*t2*u2*dt;
                fy = fy+2*v*t3*u1*du;
                fxy = fxy+6*v*t2*u1*dt*du;
                v = 4*c.c[sf+s1]-4*c.c[sf+s2]+4*c.c[sf+s3]-4*c.c[sf+s4]+2*c.c[sfx+s1]/dt+2*c.c[sfx+s2]/dt-2*c.c[sfx+s3]/dt-2*c.c[sfx+s4]/dt+2*c.c[sfy+s1]/du-2*c.c[sfy+s2]/du-2*c.c[sfy+s3]/du+2*c.c[sfy+s4]/du+1*c.c[sfxy+s1]/(dt*du)+1*c.c[sfxy+s2]/(dt*du)+1*c.c[sfxy+s3]/(dt*du)+1*c.c[sfxy+s4]/(dt*du);
                f = f+v*t3*u3;
                fx = fx+3*v*t2*u3*dt;
                fy = fy+3*v*t3*u2*du;
                fxy = fxy+9*v*t2*u2*dt*du;
                return;
            }
        }


        /*************************************************************************
        This subroutine unpacks two-dimensional spline into the coefficients table

        Input parameters:
            C   -   spline interpolant.

        Result:
            M, N-   grid size (x-axis and y-axis)
            Tbl -   coefficients table, unpacked format,
                    [0..(N-1)*(M-1)-1, 0..19].
                    For I = 0...M-2, J=0..N-2:
                        K =  I*(N-1)+J
                        Tbl[K,0] = X[j]
                        Tbl[K,1] = X[j+1]
                        Tbl[K,2] = Y[i]
                        Tbl[K,3] = Y[i+1]
                        Tbl[K,4] = C00
                        Tbl[K,5] = C01
                        Tbl[K,6] = C02
                        Tbl[K,7] = C03
                        Tbl[K,8] = C10
                        Tbl[K,9] = C11
                        ...
                        Tbl[K,19] = C33
                    On each grid square spline is equals to:
                        S(x) = SUM(c[i,j]*(x^i)*(y^j), i=0..3, j=0..3)
                        t = x-x[j]
                        u = y-y[i]

          -- ALGLIB PROJECT --
             Copyright 29.06.2007 by Bochkanov Sergey
        *************************************************************************/
        public static void spline2dunpack(spline2dinterpolant c,
            ref int m,
            ref int n,
            ref double[,] tbl)
        {
            int i = 0;
            int j = 0;
            int ci = 0;
            int cj = 0;
            int k = 0;
            int p = 0;
            int shift = 0;
            int s1 = 0;
            int s2 = 0;
            int s3 = 0;
            int s4 = 0;
            int sf = 0;
            int sfx = 0;
            int sfy = 0;
            int sfxy = 0;
            double y1 = 0;
            double y2 = 0;
            double y3 = 0;
            double y4 = 0;
            double dt = 0;
            double du = 0;

            m = 0;
            n = 0;
            tbl = new double[0,0];

            alglib.ap.assert((int)Math.Round(c.c[1])==-3 || (int)Math.Round(c.c[1])==-1, "SplineUnpack2D: incorrect C!");
            n = (int)Math.Round(c.c[2]);
            m = (int)Math.Round(c.c[3]);
            tbl = new double[(n-1)*(m-1)-1+1, 19+1];
            
            //
            // Fill
            //
            for(i=0; i<=m-2; i++)
            {
                for(j=0; j<=n-2; j++)
                {
                    p = i*(n-1)+j;
                    tbl[p,0] = c.c[4+j];
                    tbl[p,1] = c.c[4+j+1];
                    tbl[p,2] = c.c[4+n+i];
                    tbl[p,3] = c.c[4+n+i+1];
                    dt = 1/(tbl[p,1]-tbl[p,0]);
                    du = 1/(tbl[p,3]-tbl[p,2]);
                    
                    //
                    // Bilinear interpolation
                    //
                    if( (int)Math.Round(c.c[1])==-1 )
                    {
                        for(k=4; k<=19; k++)
                        {
                            tbl[p,k] = 0;
                        }
                        shift = 4+n+m;
                        y1 = c.c[shift+n*i+j];
                        y2 = c.c[shift+n*i+(j+1)];
                        y3 = c.c[shift+n*(i+1)+(j+1)];
                        y4 = c.c[shift+n*(i+1)+j];
                        tbl[p,4] = y1;
                        tbl[p,4+1*4+0] = y2-y1;
                        tbl[p,4+0*4+1] = y4-y1;
                        tbl[p,4+1*4+1] = y3-y2-y4+y1;
                    }
                    
                    //
                    // Bicubic interpolation
                    //
                    if( (int)Math.Round(c.c[1])==-3 )
                    {
                        sf = 4+n+m;
                        sfx = 4+n+m+n*m;
                        sfy = 4+n+m+2*n*m;
                        sfxy = 4+n+m+3*n*m;
                        s1 = n*i+j;
                        s2 = n*i+(j+1);
                        s3 = n*(i+1)+(j+1);
                        s4 = n*(i+1)+j;
                        tbl[p,4+0*4+0] = 1*c.c[sf+s1];
                        tbl[p,4+0*4+1] = 1*c.c[sfy+s1]/du;
                        tbl[p,4+0*4+2] = -(3*c.c[sf+s1])+3*c.c[sf+s4]-2*c.c[sfy+s1]/du-1*c.c[sfy+s4]/du;
                        tbl[p,4+0*4+3] = 2*c.c[sf+s1]-2*c.c[sf+s4]+1*c.c[sfy+s1]/du+1*c.c[sfy+s4]/du;
                        tbl[p,4+1*4+0] = 1*c.c[sfx+s1]/dt;
                        tbl[p,4+1*4+1] = 1*c.c[sfxy+s1]/(dt*du);
                        tbl[p,4+1*4+2] = -(3*c.c[sfx+s1]/dt)+3*c.c[sfx+s4]/dt-2*c.c[sfxy+s1]/(dt*du)-1*c.c[sfxy+s4]/(dt*du);
                        tbl[p,4+1*4+3] = 2*c.c[sfx+s1]/dt-2*c.c[sfx+s4]/dt+1*c.c[sfxy+s1]/(dt*du)+1*c.c[sfxy+s4]/(dt*du);
                        tbl[p,4+2*4+0] = -(3*c.c[sf+s1])+3*c.c[sf+s2]-2*c.c[sfx+s1]/dt-1*c.c[sfx+s2]/dt;
                        tbl[p,4+2*4+1] = -(3*c.c[sfy+s1]/du)+3*c.c[sfy+s2]/du-2*c.c[sfxy+s1]/(dt*du)-1*c.c[sfxy+s2]/(dt*du);
                        tbl[p,4+2*4+2] = 9*c.c[sf+s1]-9*c.c[sf+s2]+9*c.c[sf+s3]-9*c.c[sf+s4]+6*c.c[sfx+s1]/dt+3*c.c[sfx+s2]/dt-3*c.c[sfx+s3]/dt-6*c.c[sfx+s4]/dt+6*c.c[sfy+s1]/du-6*c.c[sfy+s2]/du-3*c.c[sfy+s3]/du+3*c.c[sfy+s4]/du+4*c.c[sfxy+s1]/(dt*du)+2*c.c[sfxy+s2]/(dt*du)+1*c.c[sfxy+s3]/(dt*du)+2*c.c[sfxy+s4]/(dt*du);
                        tbl[p,4+2*4+3] = -(6*c.c[sf+s1])+6*c.c[sf+s2]-6*c.c[sf+s3]+6*c.c[sf+s4]-4*c.c[sfx+s1]/dt-2*c.c[sfx+s2]/dt+2*c.c[sfx+s3]/dt+4*c.c[sfx+s4]/dt-3*c.c[sfy+s1]/du+3*c.c[sfy+s2]/du+3*c.c[sfy+s3]/du-3*c.c[sfy+s4]/du-2*c.c[sfxy+s1]/(dt*du)-1*c.c[sfxy+s2]/(dt*du)-1*c.c[sfxy+s3]/(dt*du)-2*c.c[sfxy+s4]/(dt*du);
                        tbl[p,4+3*4+0] = 2*c.c[sf+s1]-2*c.c[sf+s2]+1*c.c[sfx+s1]/dt+1*c.c[sfx+s2]/dt;
                        tbl[p,4+3*4+1] = 2*c.c[sfy+s1]/du-2*c.c[sfy+s2]/du+1*c.c[sfxy+s1]/(dt*du)+1*c.c[sfxy+s2]/(dt*du);
                        tbl[p,4+3*4+2] = -(6*c.c[sf+s1])+6*c.c[sf+s2]-6*c.c[sf+s3]+6*c.c[sf+s4]-3*c.c[sfx+s1]/dt-3*c.c[sfx+s2]/dt+3*c.c[sfx+s3]/dt+3*c.c[sfx+s4]/dt-4*c.c[sfy+s1]/du+4*c.c[sfy+s2]/du+2*c.c[sfy+s3]/du-2*c.c[sfy+s4]/du-2*c.c[sfxy+s1]/(dt*du)-2*c.c[sfxy+s2]/(dt*du)-1*c.c[sfxy+s3]/(dt*du)-1*c.c[sfxy+s4]/(dt*du);
                        tbl[p,4+3*4+3] = 4*c.c[sf+s1]-4*c.c[sf+s2]+4*c.c[sf+s3]-4*c.c[sf+s4]+2*c.c[sfx+s1]/dt+2*c.c[sfx+s2]/dt-2*c.c[sfx+s3]/dt-2*c.c[sfx+s4]/dt+2*c.c[sfy+s1]/du-2*c.c[sfy+s2]/du-2*c.c[sfy+s3]/du+2*c.c[sfy+s4]/du+1*c.c[sfxy+s1]/(dt*du)+1*c.c[sfxy+s2]/(dt*du)+1*c.c[sfxy+s3]/(dt*du)+1*c.c[sfxy+s4]/(dt*du);
                    }
                    
                    //
                    // Rescale Cij
                    //
                    for(ci=0; ci<=3; ci++)
                    {
                        for(cj=0; cj<=3; cj++)
                        {
                            tbl[p,4+ci*4+cj] = tbl[p,4+ci*4+cj]*Math.Pow(dt, ci)*Math.Pow(du, cj);
                        }
                    }
                }
            }
        }


        /*************************************************************************
        This subroutine performs linear transformation of the spline argument.

        Input parameters:
            C       -   spline interpolant
            AX, BX  -   transformation coefficients: x = A*t + B
            AY, BY  -   transformation coefficients: y = A*u + B
        Result:
            C   -   transformed spline

          -- ALGLIB PROJECT --
             Copyright 30.06.2007 by Bochkanov Sergey
        *************************************************************************/
        public static void spline2dlintransxy(spline2dinterpolant c,
            double ax,
            double bx,
            double ay,
            double by)
        {
            int i = 0;
            int j = 0;
            int n = 0;
            int m = 0;
            double v = 0;
            double[] x = new double[0];
            double[] y = new double[0];
            double[,] f = new double[0,0];
            int typec = 0;

            typec = (int)Math.Round(c.c[1]);
            alglib.ap.assert(typec==-3 || typec==-1, "Spline2DLinTransXY: incorrect C!");
            n = (int)Math.Round(c.c[2]);
            m = (int)Math.Round(c.c[3]);
            x = new double[n-1+1];
            y = new double[m-1+1];
            f = new double[m-1+1, n-1+1];
            for(j=0; j<=n-1; j++)
            {
                x[j] = c.c[4+j];
            }
            for(i=0; i<=m-1; i++)
            {
                y[i] = c.c[4+n+i];
            }
            for(i=0; i<=m-1; i++)
            {
                for(j=0; j<=n-1; j++)
                {
                    f[i,j] = c.c[4+n+m+i*n+j];
                }
            }
            
            //
            // Special case: AX=0 or AY=0
            //
            if( (double)(ax)==(double)(0) )
            {
                for(i=0; i<=m-1; i++)
                {
                    v = spline2dcalc(c, bx, y[i]);
                    for(j=0; j<=n-1; j++)
                    {
                        f[i,j] = v;
                    }
                }
                if( typec==-3 )
                {
                    spline2dbuildbicubic(x, y, f, m, n, c);
                }
                if( typec==-1 )
                {
                    spline2dbuildbilinear(x, y, f, m, n, c);
                }
                ax = 1;
                bx = 0;
            }
            if( (double)(ay)==(double)(0) )
            {
                for(j=0; j<=n-1; j++)
                {
                    v = spline2dcalc(c, x[j], by);
                    for(i=0; i<=m-1; i++)
                    {
                        f[i,j] = v;
                    }
                }
                if( typec==-3 )
                {
                    spline2dbuildbicubic(x, y, f, m, n, c);
                }
                if( typec==-1 )
                {
                    spline2dbuildbilinear(x, y, f, m, n, c);
                }
                ay = 1;
                by = 0;
            }
            
            //
            // General case: AX<>0, AY<>0
            // Unpack, scale and pack again.
            //
            for(j=0; j<=n-1; j++)
            {
                x[j] = (x[j]-bx)/ax;
            }
            for(i=0; i<=m-1; i++)
            {
                y[i] = (y[i]-by)/ay;
            }
            if( typec==-3 )
            {
                spline2dbuildbicubic(x, y, f, m, n, c);
            }
            if( typec==-1 )
            {
                spline2dbuildbilinear(x, y, f, m, n, c);
            }
        }


        /*************************************************************************
        This subroutine performs linear transformation of the spline.

        Input parameters:
            C   -   spline interpolant.
            A, B-   transformation coefficients: S2(x,y) = A*S(x,y) + B
            
        Output parameters:
            C   -   transformed spline

          -- ALGLIB PROJECT --
             Copyright 30.06.2007 by Bochkanov Sergey
        *************************************************************************/
        public static void spline2dlintransf(spline2dinterpolant c,
            double a,
            double b)
        {
            int i = 0;
            int j = 0;
            int n = 0;
            int m = 0;
            double[] x = new double[0];
            double[] y = new double[0];
            double[,] f = new double[0,0];
            int typec = 0;

            typec = (int)Math.Round(c.c[1]);
            alglib.ap.assert(typec==-3 || typec==-1, "Spline2DLinTransXY: incorrect C!");
            n = (int)Math.Round(c.c[2]);
            m = (int)Math.Round(c.c[3]);
            x = new double[n-1+1];
            y = new double[m-1+1];
            f = new double[m-1+1, n-1+1];
            for(j=0; j<=n-1; j++)
            {
                x[j] = c.c[4+j];
            }
            for(i=0; i<=m-1; i++)
            {
                y[i] = c.c[4+n+i];
            }
            for(i=0; i<=m-1; i++)
            {
                for(j=0; j<=n-1; j++)
                {
                    f[i,j] = a*c.c[4+n+m+i*n+j]+b;
                }
            }
            if( typec==-3 )
            {
                spline2dbuildbicubic(x, y, f, m, n, c);
            }
            if( typec==-1 )
            {
                spline2dbuildbilinear(x, y, f, m, n, c);
            }
        }


        /*************************************************************************
        This subroutine makes the copy of the spline model.

        Input parameters:
            C   -   spline interpolant

        Output parameters:
            CC  -   spline copy

          -- ALGLIB PROJECT --
             Copyright 29.06.2007 by Bochkanov Sergey
        *************************************************************************/
        public static void spline2dcopy(spline2dinterpolant c,
            spline2dinterpolant cc)
        {
            int n = 0;
            int i_ = 0;

            alglib.ap.assert(c.k==1 || c.k==3, "Spline2DCopy: incorrect C!");
            cc.k = c.k;
            n = (int)Math.Round(c.c[0]);
            cc.c = new double[n];
            for(i_=0; i_<=n-1;i_++)
            {
                cc.c[i_] = c.c[i_];
            }
        }


        /*************************************************************************
        Bicubic spline resampling

        Input parameters:
            A           -   function values at the old grid,
                            array[0..OldHeight-1, 0..OldWidth-1]
            OldHeight   -   old grid height, OldHeight>1
            OldWidth    -   old grid width, OldWidth>1
            NewHeight   -   new grid height, NewHeight>1
            NewWidth    -   new grid width, NewWidth>1
            
        Output parameters:
            B           -   function values at the new grid,
                            array[0..NewHeight-1, 0..NewWidth-1]

          -- ALGLIB routine --
             15 May, 2007
             Copyright by Bochkanov Sergey
        *************************************************************************/
        public static void spline2dresamplebicubic(double[,] a,
            int oldheight,
            int oldwidth,
            ref double[,] b,
            int newheight,
            int newwidth)
        {
            double[,] buf = new double[0,0];
            double[] x = new double[0];
            double[] y = new double[0];
            spline1d.spline1dinterpolant c = new spline1d.spline1dinterpolant();
            int i = 0;
            int j = 0;
            int mw = 0;
            int mh = 0;

            b = new double[0,0];

            alglib.ap.assert(oldwidth>1 && oldheight>1, "Spline2DResampleBicubic: width/height less than 1");
            alglib.ap.assert(newwidth>1 && newheight>1, "Spline2DResampleBicubic: width/height less than 1");
            
            //
            // Prepare
            //
            mw = Math.Max(oldwidth, newwidth);
            mh = Math.Max(oldheight, newheight);
            b = new double[newheight-1+1, newwidth-1+1];
            buf = new double[oldheight-1+1, newwidth-1+1];
            x = new double[Math.Max(mw, mh)-1+1];
            y = new double[Math.Max(mw, mh)-1+1];
            
            //
            // Horizontal interpolation
            //
            for(i=0; i<=oldheight-1; i++)
            {
                
                //
                // Fill X, Y
                //
                for(j=0; j<=oldwidth-1; j++)
                {
                    x[j] = (double)j/(double)(oldwidth-1);
                    y[j] = a[i,j];
                }
                
                //
                // Interpolate and place result into temporary matrix
                //
                spline1d.spline1dbuildcubic(x, y, oldwidth, 0, 0.0, 0, 0.0, c);
                for(j=0; j<=newwidth-1; j++)
                {
                    buf[i,j] = spline1d.spline1dcalc(c, (double)j/(double)(newwidth-1));
                }
            }
            
            //
            // Vertical interpolation
            //
            for(j=0; j<=newwidth-1; j++)
            {
                
                //
                // Fill X, Y
                //
                for(i=0; i<=oldheight-1; i++)
                {
                    x[i] = (double)i/(double)(oldheight-1);
                    y[i] = buf[i,j];
                }
                
                //
                // Interpolate and place result into B
                //
                spline1d.spline1dbuildcubic(x, y, oldheight, 0, 0.0, 0, 0.0, c);
                for(i=0; i<=newheight-1; i++)
                {
                    b[i,j] = spline1d.spline1dcalc(c, (double)i/(double)(newheight-1));
                }
            }
        }


        /*************************************************************************
        Bilinear spline resampling

        Input parameters:
            A           -   function values at the old grid,
                            array[0..OldHeight-1, 0..OldWidth-1]
            OldHeight   -   old grid height, OldHeight>1
            OldWidth    -   old grid width, OldWidth>1
            NewHeight   -   new grid height, NewHeight>1
            NewWidth    -   new grid width, NewWidth>1

        Output parameters:
            B           -   function values at the new grid,
                            array[0..NewHeight-1, 0..NewWidth-1]

          -- ALGLIB routine --
             09.07.2007
             Copyright by Bochkanov Sergey
        *************************************************************************/
        public static void spline2dresamplebilinear(double[,] a,
            int oldheight,
            int oldwidth,
            ref double[,] b,
            int newheight,
            int newwidth)
        {
            int i = 0;
            int j = 0;
            int l = 0;
            int c = 0;
            double t = 0;
            double u = 0;

            b = new double[0,0];

            b = new double[newheight-1+1, newwidth-1+1];
            for(i=0; i<=newheight-1; i++)
            {
                for(j=0; j<=newwidth-1; j++)
                {
                    l = i*(oldheight-1)/(newheight-1);
                    if( l==oldheight-1 )
                    {
                        l = oldheight-2;
                    }
                    u = (double)i/(double)(newheight-1)*(oldheight-1)-l;
                    c = j*(oldwidth-1)/(newwidth-1);
                    if( c==oldwidth-1 )
                    {
                        c = oldwidth-2;
                    }
                    t = (double)(j*(oldwidth-1))/(double)(newwidth-1)-c;
                    b[i,j] = (1-t)*(1-u)*a[l,c]+t*(1-u)*a[l,c+1]+t*u*a[l+1,c+1]+(1-t)*u*a[l+1,c];
                }
            }
        }


        /*************************************************************************
        Internal subroutine.
        Calculation of the first derivatives and the cross-derivative.
        *************************************************************************/
        private static void bicubiccalcderivatives(double[,] a,
            double[] x,
            double[] y,
            int m,
            int n,
            ref double[,] dx,
            ref double[,] dy,
            ref double[,] dxy)
        {
            int i = 0;
            int j = 0;
            double[] xt = new double[0];
            double[] ft = new double[0];
            double s = 0;
            double ds = 0;
            double d2s = 0;
            spline1d.spline1dinterpolant c = new spline1d.spline1dinterpolant();

            dx = new double[0,0];
            dy = new double[0,0];
            dxy = new double[0,0];

            dx = new double[m, n];
            dy = new double[m, n];
            dxy = new double[m, n];
            
            //
            // dF/dX
            //
            xt = new double[n];
            ft = new double[n];
            for(i=0; i<=m-1; i++)
            {
                for(j=0; j<=n-1; j++)
                {
                    xt[j] = x[j];
                    ft[j] = a[i,j];
                }
                spline1d.spline1dbuildcubic(xt, ft, n, 0, 0.0, 0, 0.0, c);
                for(j=0; j<=n-1; j++)
                {
                    spline1d.spline1ddiff(c, x[j], ref s, ref ds, ref d2s);
                    dx[i,j] = ds;
                }
            }
            
            //
            // dF/dY
            //
            xt = new double[m];
            ft = new double[m];
            for(j=0; j<=n-1; j++)
            {
                for(i=0; i<=m-1; i++)
                {
                    xt[i] = y[i];
                    ft[i] = a[i,j];
                }
                spline1d.spline1dbuildcubic(xt, ft, m, 0, 0.0, 0, 0.0, c);
                for(i=0; i<=m-1; i++)
                {
                    spline1d.spline1ddiff(c, y[i], ref s, ref ds, ref d2s);
                    dy[i,j] = ds;
                }
            }
            
            //
            // d2F/dXdY
            //
            xt = new double[n];
            ft = new double[n];
            for(i=0; i<=m-1; i++)
            {
                for(j=0; j<=n-1; j++)
                {
                    xt[j] = x[j];
                    ft[j] = dy[i,j];
                }
                spline1d.spline1dbuildcubic(xt, ft, n, 0, 0.0, 0, 0.0, c);
                for(j=0; j<=n-1; j++)
                {
                    spline1d.spline1ddiff(c, x[j], ref s, ref ds, ref d2s);
                    dxy[i,j] = ds;
                }
            }
        }


    }
}

