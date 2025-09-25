"""
This module contains a class that implements the main emulation method.
"""
import numpy as np
import importlib
import copy
import warnings


class emulator(object):
    def __init__(
        self,
        x=None,
        theta=None,
        f=None,
        thetaprime=None,
        method="PCGP",
        passthroughfunc=None,
        args={},
        options={},
    ):
        """
        A class used to represent an emulator or surrogate model. Fits an
        emulator or surrogate model provided in
        ``emulationmethods/[method].py`` where [method] is the user option
        with default listed above.

        .. tip::
            To use a new emulator, just drop a new file to the
            ``emulationmethods/`` directory with the required formatting.

        :Example:
            .. code-block:: python

               emulator(x=x, theta=theta, f=f, method='PCGP', args=args)

        Parameters
        ----------
        x : numpy.ndarray, optional
            An array of inputs. Each row should correspond to a row in f.
            The default is None.
            We will attempt to resolve size differences.

        theta : numpy.ndarray, optional
            An array of parameters. Each row in theta should correspond to a
            column in f. The default is None.
            We will attempt to resolve size differences.

        f : numpy.ndarray, optional
            An array of responses with 'nan' representing responses not yet
            available. The default is None.
            Each column in f should correspond to a row in x.
            Each row should correspond to a row in f.
            We will attempt to resolve if these are flipped.

        method : str, optional
            A string that points to the file located in ``emulationmethods/``.
            The default is ``PCGP``.

        passthroughfunc : function, optional
            DESCRIPTION. The default is None.

        args : dict, optional
            Optional dictionary containing options you would like to pass to
            [method].fit(x, theta, f, args)
            or
            [method].predict(x, theta, args) The default is {}.

        options : dict, optional
            Dictionary containing options you would like
            emulation to have. This does not get passed to the method.
            The default is {}.

        Returns
        -------
        None.

        """

        if ("warnings" in args.keys()) and args["warnings"]:
            warnings.resetwarnings()
        else:
            warnings.simplefilter("ignore")

        self.__ptf = passthroughfunc
        if self.__ptf is not None:
            return

        self._args = copy.deepcopy(args)

        if f is not None:
            if f.ndim < 1 or f.ndim > 2:
                raise ValueError("f must have either 1 or 2 dimensions.")
            if (x is None) and (theta is None):
                raise ValueError(
                    "You have not provided any theta or x, no"
                    " emulator inference possible."
                )
            if x is not None:
                if x.ndim < 0.5 or x.ndim > 2.5:
                    raise ValueError("x must have either 1 or 2 dimensions.")
            if theta is not None:
                if theta.ndim < 0.5 or theta.ndim > 2.5:
                    raise ValueError("theta must have either 1 or 2" " dimensions.")
        else:
            raise ValueError("You have not provided f, cannot include theta" " or x.")

        if x is not None and (f.shape[0] != x.shape[0]):
            if theta is not None:
                if (
                    f.ndim == 2
                    and f.shape[1] == x.shape[0]
                    and f.shape[0] == theta.shape[0]
                ):
                    warnings.warn("Transposing f to try to get agreement")
                    self.__f = copy.copy(f).T
                    f = copy.copy(f).T
                else:
                    raise ValueError(
                        "The number of rows in f must match the" " number of rows in x."
                    )
            else:
                if f.ndim == 2 and f.shape[1] == x.shape[0]:
                    warnings.warn("Transposing f to try to get agreement")
                    self.__f = copy.copy(f).T
                    f = copy.copy(f).T
                else:
                    raise ValueError(
                        "The number of rows in f must match the" " number of rows in x."
                    )

        if theta is not None and (f.shape[1] != theta.shape[0]):
            if x is not None:
                if not (
                    f.ndim == 2
                    and f.shape[0] == theta.shape[0]
                    and f.shape[1] == x.shape[0]
                ):
                    raise ValueError(
                        "The number of columns in f must match"
                        " the number of rows in theta."
                    )
            else:
                if f.ndim == 2 and f.shape[0] == theta.shape[0]:
                    warnings.warn("Transposing f to try to get agreement")
                    self.__f = copy.copy(f).T
                    f = copy.copy(f).T
                elif f.ndim == 1 and f.shape[0] == theta.shape[0]:
                    warnings.warn("Transposing f to try to get agreement")
                    self.__f = np.reshape(copy.copy(f), (1, -1))
                    f = np.reshape(copy.copy(f), (1, -1))
                else:
                    raise ValueError(
                        "The number of columns in f must match"
                        " the number of rows in theta."
                    )

        if x is not None:
            self.__x = copy.copy(x)
        else:
            self.__x = None

        if theta is not None:
            self.__theta = copy.copy(theta)
        else:
            self.__theta = None
            raise ValueError("This feature has not developed yet.")

        self.__suppx = None
        self.__supptheta = None
        self.__f = copy.copy(f)

        try:
            self.method = importlib.import_module("PUQ.surrogatemethods." + method)
        except Exception:
            raise ValueError("Module not loaded correctly.")
        if "fit" not in dir(self.method):
            raise ValueError("Function fit not found in module!")
        if "predict" not in dir(self.method):
            raise ValueError("Function predict not found in module!")
        if "supplementtheta" not in dir(self.method):
            warnings.warn("Function supplementtheta not found in module!")

        self.__options = {}
        self.__optionsset(options)
        self._info = {}
        self._info = {"method": method}

        if (self.__f is not None) and (self.__options["autofit"]):
            self.fit()

    def __repr__(self):
        object_method = [
            method_name
            for method_name in dir(self)
            if callable(getattr(self, method_name))
        ]
        object_method = [x for x in object_method if not x.startswith("__")]
        strrepr = (
            "An emulation object where the code in located in the file "
            + "emulation. The main method are emu."
            + ", emu.".join(object_method)
            + ". Default of emu(x,theta)"
            " is emu.predict(x,theta). "
            "Run help(emu) for the document string."
        )
        return strrepr

    def __call__(self, x=None, theta=None, thetaprime=None, args=None):
        return self.predict(x, theta, thetaprime, args)

    def fit(self, args=None):
        """
        Fits an emulator or surrogate and places that in emu._info

        Calls
        emu._info = [method].fit(emu.__theta, emu.__f, emu.__x, args = args)

        Parameters
        ----------
        args : dict
            Optional dictionary containing options you would like to pass to
            fit function. It will add/modify those in self._args.
        """
        if args is not None:
            argstemp = {**self._args, **copy.deepcopy(args)}
        else:
            argstemp = copy.copy(self._args)
        x, theta, f = self.__preprocess()

        self.method.fit(self._info, x, theta, f, **argstemp)

    def predict(self, x=None, theta=None, thetaprime=None, args={}):
        """
        Fits an emulator or surrogate.

        :Example:
            .. code-block:: python

              emulator.predict(x=x, theta=theta, args=args)

        Parameters
        ----------
        x : numpy.ndarray, optional
            An array of inputs. Each row in x should correspond to a row in f.
            The default is None.

        theta : numpy.ndarray, optional
            An array of parameters. Each row should correspond to a
            column in f. The default is None.

        args : dict, optional
            A dictionary containing args. The default is {}.

        Raises
        ------
        ValueError
            If the dimensions of inputs do not match with the fitted
            emulator.

        Returns
        -------
        surmise.emulation.prediction
            An instance of emulation class prediction

        """

        if self.__ptf is not None:
            info = {}
            if theta is not None:
                info["mean"] = self.__ptf(x, theta)
            else:
                info["mean"] = self.__ptf(x, self.__theta)
            info["var"] = 0 * info["mean"]
            info["covxhalf"] = 0 * np.stack((info["mean"], info["mean"]), 2)
            return prediction(info, self)
        if args is not None:
            argstemp = {**self._args, **copy.deepcopy(args)}
        else:
            argstemp = copy.copy(self._args)
        if x is None:
            x = copy.copy(self.__x)
        else:
            x = copy.copy(x)
            if x.ndim == 1:
                if self.__x.ndim == 2 and x.shape[0] == self.__x.shape[1]:
                    x = np.reshape(x, (1, -1))
                elif self.__x.ndim == 2:
                    raise ValueError(
                        "Your x shape seems to not agree with the" " emulator build."
                    )
            elif x.ndim == 2:
                if self.__x.ndim == 1:
                    raise ValueError(
                        "Your x shape seems to not agree with the" " emulator build."
                    )
                elif (
                    x.shape[1] != self.__x.shape[1] and x.shape[0] == self.__x.shape[1]
                ):
                    x = x.T
                elif (
                    x.shape[1] != self.__x.shape[1] and x.shape[0] != self.__x.shape[1]
                ):
                    raise ValueError(
                        "Your x shape seems to not agree with the" " emulator build."
                    )
        if theta is None:
            theta = copy.copy(self.__theta)
        else:
            theta = copy.copy(theta)
            if theta.ndim == 2 and self.__theta.ndim == 1:
                raise ValueError(
                    "Your theta shape seems to not agree with the" " emulator build."
                )
            # note: dont understand why we have this statement
            elif (
                theta.ndim == 1
                and self.__theta.ndim == 2
                and theta.shape[0] == self.__theta.shape[1]
            ):
                theta = np.reshape(theta, (1, -1))
            elif theta.ndim == 1 and self.__theta.ndim == 2:
                raise ValueError(
                    "Your theta shape seems to not agree with the" " emulator build."
                )
            elif (
                theta.shape[1] != self.__theta.shape[1]
                and theta.shape[0] == self.__theta.shape[1]
            ):
                theta = theta.T
            elif (
                theta.shape[1] != self.__theta.shape[1]
                and theta.shape[0] != self.__theta.shape[1]
            ):
                raise ValueError(
                    "Your theta shape seems to not agree with the" " emulator build."
                )

        info = {}
        self.method.predict(info, self._info, x, theta, thetaprime, **argstemp)
        return prediction(info, self)
    def update(self,x=None,Y=None,**kwargs):

        self.method.update(self._info,x=x,Y=Y,**kwargs)

    def acquisition(self, x=None, theta1=None, theta2=None):
        return self.method.acquisition(self._info, x, theta1, theta2)

    def computeC(self, x=None, theta1=None, realdata=None, realvar=None):

        return self.method.computeC(self._info, x, theta1, realdata, realvar)
        


    def __optionsset(self, options=None):
        options = copy.deepcopy(options)
        # options will always be lowercase
        options = {k.lower(): v for k, v in options.items()}

        if "thetareps" in options.keys():
            if type(options["thetareps"]) is bool:
                self.__options["thetareps"] = options["thetareps"]
            else:
                raise ValueError("option thetareps must be true or false")

        if "xreps" in options.keys():
            if type(options["xreps"]) is bool:
                self.__options["xreps"] = options["xreps"]
            else:
                raise ValueError("option xreps must be true or false")

        if "thetarmnan" in options.keys():
            if type(options["thetarmnan"]) is bool:
                if options["thetarmnan"]:
                    self.__options["thetarmnan"] = 0
                else:
                    self.__options["thetarmnan"] = 1 + (10 ** (-12))
            elif type(options["thetarmnan"]) is str:
                if (
                    isinstance(options["thetarmnan"], str)
                    and options["thetarmnan"] == "any"
                ):
                    self.__options["thetarmnan"] = 0
                elif (
                    isinstance(options["thetarmnan"], str)
                    and options["thetarmnan"] == "some"
                ):
                    self.__options["thetarmnan"] = 0.2
                elif (
                    isinstance(options["thetarmnan"], str)
                    and options["thetarmnan"] == "most"
                ):
                    self.__options["thetarmnan"] = 0.5
                elif (
                    isinstance(options["thetarmnan"], str)
                    and options["thetarmnan"] == "alot"
                ):
                    self.__options["thetarmnan"] = 0.8
                elif (
                    isinstance(options["thetarmnan"], str)
                    and options["thetarmnan"] == "all"
                ):
                    self.__options["thetarmnan"] = 1 - (10 ** (-8))
                elif (
                    isinstance(options["thetarmnan"], str)
                    and options["thetarmnan"] == "never"
                ):
                    self.__options["thetarmnan"] = 1 + (10 ** (-8))
                else:
                    raise ValueError(
                        "option thetarmnan must be True, False,"
                        " "
                        "any"
                        ", "
                        "some"
                        ", "
                        "most"
                        ", "
                        "alot"
                        ","
                        " "
                        "all"
                        ", "
                        "never"
                        " or an scaler bigger"
                        "than zero and less than one."
                    )
            elif (
                np.isfinite(options["thetarmnan"])
                and options["thetarmnan"] >= 0
                and options["thetarmnan"] <= 1
            ):
                self.__options["thetarmnan"] = options["thetarmnan"]
            else:
                raise ValueError(
                    "option thetarmnan must be True, False,"
                    " "
                    "any"
                    ", "
                    "some"
                    ", "
                    "most"
                    ", "
                    "alot"
                    ","
                    " "
                    "all"
                    ", "
                    "never"
                    " or an scaler bigger"
                    "than zero and less than one."
                )
        if "xrmnan" in options.keys():
            if type(options["xrmnan"]) is bool:
                if options["xrmnan"]:
                    self.__options["xrmnan"] = 0
                else:
                    self.__options["xrmnan"] = 1 + (10 ** (-12))
            elif type(options["xrmnan"]) is str:
                if isinstance(options["xrmnan"], str) and options["xrmnan"] == "any":
                    self.__options["xrmnan"] = 0
                elif isinstance(options["xrmnan"], str) and options["xrmnan"] == "some":
                    self.__options["xrmnan"] = 0.2
                elif isinstance(options["xrmnan"], str) and options["xrmnan"] == "most":
                    self.__options["xrmnan"] = 0.5
                elif isinstance(options["xrmnan"], str) and options["xrmnan"] == "alot":
                    self.__options["xrmnan"] = 0.8
                elif isinstance(options["xrmnan"], str) and options["xrmnan"] == "all":
                    self.__options["xrmnan"] = 1 - (10 ** (-8))
                elif (
                    isinstance(options["xrmnan"], str) and options["xrmnan"] == "never"
                ):
                    self.__options["xrmnan"] = 1 + (10 ** (-8))
                else:
                    raise ValueError(
                        "option xrmnan must be True, False,"
                        " "
                        "any"
                        ", "
                        "some"
                        ", "
                        "most"
                        ", "
                        "alot"
                        ","
                        " "
                        "all"
                        ", "
                        "never"
                        " or an scaler bigger"
                        "than zero and less than one."
                    )
            elif (
                np.isfinite(options["xrmnan"])
                and options["xrmnan"] >= 0
                and options["xrmnan"] <= 1
            ):
                self.__options["xrmnan"] = options["xrmnan"]
            else:
                raise ValueError(
                    "option xrmnan must be True, False,"
                    " "
                    "any"
                    ", "
                    "some"
                    ", "
                    "most"
                    ", "
                    "alot"
                    ","
                    " "
                    "all"
                    ", "
                    "never"
                    " or an scaler bigger"
                    "than zero and less than one."
                )

        if "rmthetafirst" in options.keys():
            if type(options["rmthetafirst"]) is bool:
                self.__options["rmthetafirst"] = options["rmthetafirst"]
            else:
                raise ValueError("option rmthetafirst must be True or False.")

        if "autofit" in options.keys():
            if type(options["autofit"]) is bool:
                self.__options["minsampsize"] = options["autofit"]
            else:
                raise ValueError("option autofit must be of type bool.")

        if "thetareps" not in self.__options.keys():
            self.__options["thetareps"] = False
        if "xreps" not in self.__options.keys():
            self.__options["xreps"] = False
        if "thetarmnan" not in self.__options.keys():
            self.__options["thetarmnan"] = 0.8
        if "xrmnan" not in self.__options.keys():
            self.__options["xrmnan"] = 0.8
        if "autofit" not in self.__options.keys():
            self.__options["autofit"] = True
        if "rmthetafirst" not in self.__options.keys():
            self.__options["rmthetafirst"] = True

    def __preprocess(self):
        x = copy.copy(self.__x)
        theta = copy.copy(self.__theta)
        f = copy.copy(self.__f)
        options = self.__options
        isinff = np.isinf(f)
        if np.any(isinff):
            print("All infs were converted to nans.")
            f[isinff] = float("NaN")
        isnanf = np.isnan(f)

        # first, check missing thetas
        if self.__options["rmthetafirst"]:
            j = np.where(np.mean(isnanf, 0) < options["thetarmnan"])[0]
            f = f[:, j]
            if theta.ndim == 1:
                theta = theta[j]
            else:
                theta = theta[j, :]
        # then, check missing xs
        j = np.where(np.mean(isnanf, 1) < options["xrmnan"])[0]
        f = f[j, :]
        if x is not None:
            if x.ndim == 1:
                x = x[j]
            else:
                x = x[j, :]

        if not self.__options["rmthetafirst"]:
            j = np.where(np.mean(isnanf, 0) < options["thetarmnan"])[0]
            f = f[:, j]
            if theta.ndim == 1:
                theta = theta[j]
            else:
                theta = theta[j, :]
        return x, theta, f


class prediction(object):
    """
    A class to represent an emulation prediction. predict._info returns the
    dictionary from the method.

        :Example:

            .. code-block:: python

                prediction.mean()

                prediction.var()

                prediction.covx()

                prediction.rnd()
    """

    def __init__(self, _info, emu):
        self._info = _info
        self.emu = emu

    def __repr__(self):
        object_method = [
            method_name
            for method_name in dir(self)
            if callable(getattr(self, method_name))
        ]
        object_method = [x for x in object_method if not x.startswith("_")]
        object_method = [x for x in object_method if not x.startswith("emu")]
        strrepr = (
            "A emulation prediction object predict where the code in"
            " located in the file "
            + " emulation.  The main method are predict."
            + ", predict.".join(object_method)
            + ".  Default of predict()"
            " is predict.mean() and " + "predict(s) will run pred.rnd(s)."
            " Run help(predict) for the document" + " string."
        )
        return strrepr

    def __call__(self, s=None, args=None):
        if s is None:
            return self.mean(args)
        else:
            return self.rnd(s, args)

    def __methodnotfoundstr(self, pfstr, opstr):
        msg = (
            pfstr
            + opstr
            + " functionality not in method... \n"
            + " Key labeled "
            + opstr
            + " not "
            + "provided in "
            + pfstr
            + "._info... \n"
            + " Key labeled rnd not "
            + "provided in "
            + pfstr
            + "._info..."
        )

        return msg

    def mean(self, args=None):
        """
        Returns the mean at theta and x in when building the prediction.
        """

        pfstr = "predict"  # prefix string
        opstr = "mean"  # operation string
        if (self.emu._emulator__ptf is None) and (
            (pfstr + opstr) in dir(self.emu.method)
        ):
            if args is None:
                args = self.emu._args
            return copy.deepcopy(self.emu.method.predictmean(self._info, **args))
        elif opstr in self._info.keys():
            return self._info[opstr]
        elif "rnd" in self._info.keys():
            return copy.deepcopy(np.mean(self._info["rnd"], 0))
        else:
            raise ValueError(self.__methodnotfoundstr(pfstr, opstr))

    def var(self, args=None):
        """
        Returns the pointwise variance at theta and x when building
        the prediction.
        """

        pfstr = "predict"  # prefix string
        opstr = "var"  # operation string
        if (self.emu._emulator__ptf is None) and (
            (pfstr + opstr) in dir(self.emu.method)
        ):
            if args is None:
                args = self.emu._args
            return copy.deepcopy(self.emu.method.predictvar(self._info, **args))
        elif opstr in self._info.keys():
            return copy.deepcopy(self._info[opstr])
        elif "rnd" in self._info.keys():
            return copy.deepcopy(np.var(self._info["rnd"], 0))
        else:
            raise ValueError(self.__methodnotfoundstr(pfstr, opstr))
