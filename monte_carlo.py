import random
import ctypes
import random
from ctypes import c_float, c_int32, POINTER

# Run `odin build monte-carlo.odin -file -build-mode:dll -o:speed`
# https://odin-lang.org/news/calling-odin-from-python/
# https://stackoverflow.com/a/51132594

lib = ctypes.CDLL("./monte-carlo.dll")

class Polynomial:
    def __init__(self, terms = None):
        # We store each term as an entry in the dictionary as ((var1, deg), (var2, deg), ...): coefficient,
        # where var1 index < var2 index < ...
        # For example, x0**2x2 - 2x0x1 + 1 will be { ((0, 2), (2, 1)): 1, ((0, 1), (1, 1)): -2, (): 1 }.
        if terms == None:
            terms = {}
        self.terms = terms.copy()
    
    def __neg__(self):
        terms = self.terms.copy()
        for variables in terms:
            terms[variables] *= -1
        return Polynomial(terms)
    
    def __add__(self, other):
        if type(other) == int or type(other) is float:
            other = constant_polynomial(other)
        assert type(other) == Polynomial

        terms = self.terms.copy()
        for (variables, coefficient) in other.terms.items():
            if variables in terms:
                terms[variables] += coefficient
                if abs(terms[variables]) < 1e-9:
                    del terms[variables]
            else:
                terms[variables] = coefficient
        
        return Polynomial(terms)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if type(other) == int or type(other) is float:
            other = constant_polynomial(other)
        assert type(other) == Polynomial

        terms = self.terms.copy()
        for (variables, coefficient) in other.terms.items():
            if variables in terms:
                terms[variables] -= coefficient
                if abs(terms[variables]) < 1e-9:
                    del terms[variables]
            else:
                terms[variables] = -coefficient
        
        return Polynomial(terms)

    def __rsub__(self, other):
       return -(self - other)

    def __mul__(self, other):
        if type(other) == int or type(other) is float:
            other = constant_polynomial(other)
        assert type(other) == Polynomial

        terms = {}
        for (variables1, coefficient1) in self.terms.items():
            for (variables2, coefficient2) in other.terms.items():
                coefficient = coefficient1 * coefficient2
                variables = {}
                for (variable, degree) in variables1:
                    variables[variable] = degree
                for (variable, degree) in variables2:
                    if variable in variables:
                        variables[variable] += degree
                    else:
                        variables[variable] = degree
                variables = tuple(sorted(list(variables.items())))
                
                if variables in terms:
                    terms[variables] += coefficient
                else:
                    terms[variables] = coefficient

        terms_keys = list(terms.keys())
        for key in terms_keys:
            if terms[key] == 0:
                del terms[key]
        
        return Polynomial(terms)

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        res = constant_polynomial(1)
        for _ in range(other):
            res *= self
        return res

    def __repr__(self):
        return " + ".join([
            f"{coefficient if coefficient != 1 or len(variables) == 0 else ""}" + "".join([
                f"x{index}{f"^{degree}" if degree != 1 else ""}" for (index, degree) in variables
            ])
            for (variables, coefficient) in self.terms.items()
        ])
    
    def number_of_variables(self):
        res = 0
        for variables in self.terms:
            for (index, _) in variables:
                res = max(res, index + 1)
        return res

    def __call__(self, val):
        if type(val) == float or type(val) == int:
            val = (val,)
        assert len(val) == self.number_of_variables()

        res = 0
        for (variables, coefficient) in self.terms.items():
            prod = coefficient
            for (index, degree) in variables:
                prod *= val[index] ** degree
            res += prod
        return res

def constant_polynomial(val):
    return Polynomial({(): val})

def single_variable_polynomial(index):
    return Polynomial({((index, 1),): 1})

def polynomial_integral(polynomial: Polynomial, region):
    if type(region[0]) == int or type(region[0]) == float:
        region = [region]
    n = polynomial.number_of_variables()
    assert n == len(region)

    res = 0
    for (variables, coefficient) in polynomial.terms.items():
        degrees = [0] * n
        for (index, degree) in variables:
            degrees[index] = degree
        
        integral = coefficient
        for i in range(n):
            integral *= region[i][1] ** (degrees[i] + 1) / (degrees[i] + 1) - region[i][0] ** (degrees[i] + 1) / (degrees[i] + 1)
        res += integral
    
    return res

def grid_integral(f, region, n = 10, cell_offset = 0.0):
    if type(region[0]) == int or type(region[0]) == float:
        region = [region]

    counter = [0] * len(region)
    res = 0
    cell_volume = 1
    for (a, b) in region:
        cell_volume *= (b - a) / n

    loop = True
    while loop:
        res += f([region[i][0] + (counter[i] + cell_offset) / n * (region[i][1] - region[i][0]) for i in range(len(counter))]) * cell_volume
        loop = False
        for i in range(len(counter) - 1, -1, -1):
            if counter[i] == n - 1:
                counter[i] = 0
            else:
                counter[i] += 1
                loop = True
                break
    return res

def monte_carlo_integral(f, region, n = 10):
    if type(region[0]) == int or type(region[0]) == float:
        region = [region]

    res = 0
    for _ in range(n):
        x = [region[i][0] + random.random() * (region[i][1] - region[i][0]) for i in range(len(region))]
        res += f(x)
    res /= n

    volume = 1
    for (a, b) in region:
        volume *= (b - a)
    res *= volume

    return res

lib.polynomial_monte_carlo_integral.argtypes = [c_int32, POINTER(c_int32), POINTER(c_float), POINTER(c_float), c_int32]
lib.polynomial_monte_carlo_integral.restype = c_float

def polynomial_monte_carlo_integral(polynomial: Polynomial, region, n):
    if type(region[0]) == int or type(region[0]) == float:
        region = [region]
    assert polynomial.number_of_variables() == len(region)

    max_degrees = [0] * polynomial.number_of_variables()
    for term in polynomial.terms:
        for index, degree in term:
            max_degrees[index] = max(degree, max_degrees[index])

    number_of_coefficients = 1
    for max_degree in max_degrees:
        number_of_coefficients *= max_degree + 1
    index_multiplier = [1] * len(max_degrees)
    for i in range(len(max_degrees) - 2, -1, -1):
        index_multiplier[i] = index_multiplier[i+1] * (max_degrees[i+1] + 1)
    coefficients = [0] * number_of_coefficients
    for term in polynomial.terms:
        coefficient_index = 0
        for index, degree in term:
            coefficient_index += index_multiplier[index] * degree
        coefficients[coefficient_index] = polynomial.terms[term]

    return lib.polynomial_monte_carlo_integral(
        len(max_degrees),
        (c_int32 * len(max_degrees))(*max_degrees),
        (c_float * len(coefficients))(*coefficients),
        (c_float * (2 *len(region)))(*sum(region, start=())),
        n
    )

lib.polynomial_grid_midpoint_integral.argtypes = [c_int32, POINTER(c_int32), POINTER(c_float), POINTER(c_float), c_int32]
lib.polynomial_grid_midpoint_integral.restype = c_float

def polynomial_grid_midpoint_integral(polynomial: Polynomial, region, n):
    if type(region[0]) == int or type(region[0]) == float:
        region = [region]
    assert polynomial.number_of_variables() == len(region)

    max_degrees = [0] * polynomial.number_of_variables()
    for term in polynomial.terms:
        for index, degree in term:
            max_degrees[index] = max(degree, max_degrees[index])

    number_of_coefficients = 1
    for max_degree in max_degrees:
        number_of_coefficients *= max_degree + 1
    index_multiplier = [1] * len(max_degrees)
    for i in range(len(max_degrees) - 2, -1, -1):
        index_multiplier[i] = index_multiplier[i+1] * (max_degrees[i+1] + 1)
    coefficients = [0] * number_of_coefficients
    for term in polynomial.terms:
        coefficient_index = 0
        for index, degree in term:
            coefficient_index += index_multiplier[index] * degree
        coefficients[coefficient_index] = polynomial.terms[term]

    return lib.polynomial_grid_midpoint_integral(
        len(max_degrees),
        (c_int32 * len(max_degrees))(*max_degrees),
        (c_float * len(coefficients))(*coefficients),
        (c_float * (2 *len(region)))(*sum(region, start=())),
        n
    )