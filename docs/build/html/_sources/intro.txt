A Brief Introduction to pyFlare
===============================

The development of pyFlare was driven by the need for an automated
means to identify flaring events in lightcurves released by the Kepler
mission. This has led to the modern package containing logic to
perform hypothesis testing and produce detection statistics, in
addition to being able to handle the large quantities of data which
are available, and the additional data required to successfully run an
analysis.

The statistical methods used in pyFlare owe much to data analysis
developments from the field of gravitational wave research; the
detection statistic which is used is based on one developed to
identify ring downs in gravtitational wave detector data.

The power of this technique has increased as it was developed, and the
package now contains logic capable of identifying planetary transits
in the same Kepler data, and can be extended to perform searches for
arbitrarily defined profiles.

