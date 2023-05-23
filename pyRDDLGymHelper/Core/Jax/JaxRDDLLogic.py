import jax
import jax.numpy as jnp
import jax.random as random
import warnings

from typing import Set


class Complement:
    
    def __call__(self, x):
        raise NotImplementedError


class StandardComplement(Complement):
    
    def __call__(self, x):
        return 1.0 - x


class TNorm:
    
    def norm(self, x, y):
        raise NotImplementedError
    
    def norms(self, x, axis):
        raise NotImplementedError
        

class ProductTNorm(TNorm):
    
    def norm(self, x, y):
        return x * y
    
    def norms(self, x, axis):
        return jnp.prod(x, axis=axis)


class GodelTNorm(TNorm):
    
    def norm(self, x, y):
        return jnp.minimum(x, y)
    
    def norms(self, x, axis):
        return jnp.min(x, axis=axis)


class LukasiewiczTNorm(TNorm):
    
    def norm(self, x, y):
        return jax.nn.relu(x + y - 1.0)
    
    def norms(self, x, axis):
        return jax.nn.relu(jnp.sum(x - 1.0, axis=axis) + 1.0)
    
    
class FuzzyLogic:
    '''A class representing fuzzy logic in JAX.
    
    Functionality can be customized by either providing a tnorm as parameters, 
    or by overriding its methods.
    '''
    
    def __init__(self, tnorm: TNorm=ProductTNorm(),
                 complement: Complement=StandardComplement(),
                 weight: float=10.0,
                 debias: Set[str]={},
                 eps: float=1e-10):
        '''Creates a new fuzzy logic in Jax.
        
        :param tnorm: fuzzy operator for logical AND
        :param complement: fuzzy operator for logical NOT
        :param weight: a concentration parameter (larger means better accuracy)
        :param error: an error parameter (e.g. floor) (smaller means better accuracy)
        :param debias: which functions to de-bias approximate on forward pass
        :param eps: small positive float to mitigate underflow
        '''
        self.tnorm = tnorm
        self.complement = complement
        self.weight = float(weight)
        self.debias = debias
        self.eps = eps
        
    # ===========================================================================
    # logical operators
    # ===========================================================================
     
    def And(self):
        warnings.warn('Using the replacement rule: '
                      'a ^ b --> tnorm(a, b).', stacklevel=2)
        
        _and = self.tnorm.norm
        
        def _jax_wrapped_calc_and_approx(a, b, param):
            return _and(a, b)
        
        return _jax_wrapped_calc_and_approx, None
    
    def Not(self):
        warnings.warn('Using the replacement rule: '
                      '~a --> 1 - a', stacklevel=2)
        
        _not = self.complement
        
        def _jax_wrapped_calc_not_approx(x, param):
            return _not(x)
        
        return _jax_wrapped_calc_not_approx, None
    
    def Or(self):
        warnings.warn('Using the replacement rule: '
                      'a or b --> tconorm(a, b).', stacklevel=2)
        
        _not = self.complement
        _and = self.tnorm.norm
        
        def _jax_wrapped_calc_or_approx(a, b, param):
            return _not(_and(_not(a), _not(b)))
        
        return _jax_wrapped_calc_or_approx, None

    def xor(self):
        warnings.warn('Using the replacement rule: '
                      'a xor b --> (a or b) ^ (a ^ b).', stacklevel=2)
        
        _not = self.complement
        _and = self.tnorm.norm
        
        def _jax_wrapped_calc_xor_approx(a, b, param):
            _or = _not(_and(_not(a), _not(b)))
            return _and(_or(a, b), _not(_and(a, b)))
        
        return _jax_wrapped_calc_xor_approx, None
        
    def implies(self):
        warnings.warn('Using the replacement rule: '
                      'a => b --> ~a ^ b', stacklevel=2)
        
        _not = self.complement
        _and = self.tnorm.norm
        
        def _jax_wrapped_calc_implies_approx(a, b, param):
            return _not(_and(a, _not(b)))
        
        return _jax_wrapped_calc_implies_approx, None
    
    def equiv(self):
        warnings.warn('Using the replacement rule: '
                      'a <=> b --> (a => b) ^ (b => a)', stacklevel=2)
        
        _not = self.complement
        _and = self.tnorm.norm
        
        def _jax_wrapped_calc_equiv_approx(a, b, param):
            atob = _not(_and(a, _not(b)))
            btoa = _not(_and(b, _not(a)))
            return _and(atob, btoa)
        
        return _jax_wrapped_calc_equiv_approx, None
    
    def forall(self):
        warnings.warn('Using the replacement rule: '
                      'forall(a) --> tnorm(a[1], tnorm(a[2], ...))', stacklevel=2)
        
        _forall = self.tnorm.norms
        
        def _jax_wrapped_calc_forall_approx(x, axis, param):
            return _forall(x, axis=axis)
        
        return _jax_wrapped_calc_forall_approx, None
    
    def exists(self):
        _not = self.complement
        jax_forall, jax_param = self.forall()
        
        def _jax_wrapped_calc_exists_approx(x, axis, param):
            return _not(jax_forall(_not(x), axis, param))
        
        return _jax_wrapped_calc_exists_approx, jax_param
    
    # ===========================================================================
    # comparison operators
    # ===========================================================================
     
    def greaterEqual(self):
        warnings.warn('Using the replacement rule: '
                      'a >= b --> sigmoid(a - b)', stacklevel=2)
        debias = 'greaterEqual' in self.debias
        
        def _jax_wrapped_calc_geq_approx(a, b, param):
            sample = jax.nn.sigmoid(param * (a - b))
            if debias:
                hard_sample = jnp.greater_equal(a, b)
                sample += jax.lax.stop_gradient(hard_sample - sample)
            return sample
        
        tags = ('weight', 'greaterEqual')
        new_param = (tags, self.weight)
        return _jax_wrapped_calc_geq_approx, new_param
    
    def greater(self):
        warnings.warn('Using the replacement rule: '
                      'a > b --> sigmoid(a - b)', stacklevel=2)
        debias = 'greater' in self.debias
        
        def _jax_wrapped_calc_gre_approx(a, b, param):
            sample = jax.nn.sigmoid(param * (a - b))
            if debias:
                hard_sample = jnp.greater(a, b)
                sample += jax.lax.stop_gradient(hard_sample - sample)
            return sample
        
        tags = ('weight', 'greater')
        new_param = (tags, self.weight)
        return _jax_wrapped_calc_gre_approx, new_param
    
    def lessEqual(self):
        jax_geq, jax_param = self.greaterEqual()
        
        def _jax_wrapped_calc_leq_approx(a, b, param):
            return jax_geq(-a, -b, param)
        
        return _jax_wrapped_calc_leq_approx, jax_param
    
    def less(self):
        jax_gre, jax_param = self.greater()

        def _jax_wrapped_calc_less_approx(a, b, param):
            return jax_gre(-a, -b, param)
        
        return _jax_wrapped_calc_less_approx, jax_param

    def equal(self):
        warnings.warn('Using the replacement rule: '
                      'a == b --> sech^2(b - a)', stacklevel=2)
        debias = 'equal' in self.debias
        
        def _jax_wrapped_calc_equal_approx(a, b, param):
            sample = 1.0 - jnp.square(jnp.tanh(param * (b - a)))
            if debias:
                hard_sample = jnp.equal(a, b)
                sample += jax.lax.stop_gradient(hard_sample - sample)
            return sample
        
        tags = ('weight', 'equal')
        new_param = (tags, self.weight)
        return _jax_wrapped_calc_equal_approx, new_param
    
    def notEqual(self):
        _not = self.complement
        jax_eq, jax_param = self.equal()
        
        def _jax_wrapped_calc_neq_approx(a, b, param):
            return _not(jax_eq(a, b, param))

        return _jax_wrapped_calc_neq_approx, jax_param
        
    # ===========================================================================
    # special functions
    # ===========================================================================
     
    def signum(self):
        warnings.warn('Using the replacement rule: '
                      'signum(x) --> tanh(x)', stacklevel=2)
        debias = 'signum' in self.debias
        
        def _jax_wrapped_calc_signum_approx(x, param):
            sample = jnp.tanh(param * x)
            if debias:
                hard_sample = jnp.sign(x)
                sample += jax.lax.stop_gradient(hard_sample - sample)
            return sample
        
        tags = ('weight', 'signum')
        new_param = (tags, self.weight)
        return _jax_wrapped_calc_signum_approx, new_param
    
    # def _sawtooth(self, x, d):
    #     pi = jnp.pi
    #     trg = 1.0 - 2.0 * jnp.arccos((1.0 - d) * jnp.sin(pi * (x - 0.5))) / pi
    #     sqr = 2.0 * jnp.arctan(jnp.sin(pi * x) / d) / pi
    #     swt = (1.0 + trg * sqr) / 2.0
    #     return swt        
    #
    # def floor(self):
    #     warnings.warn('Using the replacement rule: '
    #                   'floor(x) --> x - sawtooth(x), where sawtooth is a '
    #                   'trigonometric approximation of the sawtooth function',
    #                   stacklevel=2)
    #     debias = 'floor' in self.debias
    #
    #     def _jax_wrapped_calc_floor_approx(x, param):
    #         sample = x - self._sawtooth(x, param)
    #         if debias:
    #             hard_sample = jnp.floor(x)
    #             sample += jax.lax.stop_gradient(hard_sample - sample)
    #         return sample
    #
    #     tags = ('error', 'floor')
    #     new_param = (tags, self.error)
    #     return _jax_wrapped_calc_floor_approx, new_param
    #
    # def ceil(self):
    #     jax_floor, jax_param = self.floor()
    #
    #     def _jax_wrapped_calc_ceil_approx(x, param):
    #         return -jax_floor(-x, param)
    #
    #     return _jax_wrapped_calc_ceil_approx, jax_param
    
    def ceil(self):
        warnings.warn('Using the replacement rule: '
                      'ceil(x) --> ceil(x - 0.5) + step(x - 0.5), '
                      'where step is a smooth approximation of the step function')
        debias = 'ceil' in self.debias
        
        def _jax_wrapped_calc_ceil_approx(x, param):
            diff = (x - 0.5) - jnp.floor(x - 0.5)
            arg = jnp.log1p((2.0 * diff - 1.0) / (1.0 - diff) + self.eps)            
            sample = jnp.ceil(x - 0.5) + jax.nn.sigmoid(arg * param)
            if debias:
                hard_sample = jnp.ceil(x)
                sample += jax.lax.stop_gradient(hard_sample - sample)
            return sample
        
        tags = ('weight', 'ceil')
        new_param = (tags, self.weight)
        return _jax_wrapped_calc_ceil_approx, new_param
    
    def floor(self):
        jax_ceil, jax_param = self.ceil()
        
        def _jax_wrapped_calc_floor_approx(x, param):
            return -jax_ceil(-x, param)
        
        return _jax_wrapped_calc_floor_approx, jax_param
        
    def round(self):
        warnings.warn('Using the replacement rule: '
                      'round(x) --> x', stacklevel=2)
        
        def _jax_wrapped_calc_round_approx(x, param):
            return x
        
        return _jax_wrapped_calc_round_approx, None
    
    def mod(self):
        jax_floor, jax_param = self.floor()
        
        def _jax_wrapped_calc_mod_approx(x, y, param):
            return x - y * jax_floor(x / y, param)
        
        return _jax_wrapped_calc_mod_approx, jax_param
    
    def floorDiv(self):
        jax_floor, jax_param = self.floor()
        
        def _jax_wrapped_calc_mod_approx(x, y, param):
            return jax_floor(x / y, param)
        
        return _jax_wrapped_calc_mod_approx, jax_param
    
    def sqrt(self):
        warnings.warn('Using the replacement rule: '
                      'sqrt(x) --> sqrt(x + eps)', stacklevel=2)
        
        def _jax_wrapped_calc_sqrt_approx(x, param):
            return jnp.sqrt(x + self.eps)
        
        return _jax_wrapped_calc_sqrt_approx, None
    
    # ===========================================================================
    # indexing
    # ===========================================================================
     
    @staticmethod
    def _literals(shape, axis):
        literals = jnp.arange(shape[axis])
        literals = literals[(...,) + (jnp.newaxis,) * (len(shape) - 1)]
        literals = jnp.moveaxis(literals, source=0, destination=axis)
        literals = jnp.broadcast_to(literals, shape=shape)
        return literals
    
    def argmax(self):
        warnings.warn('Using the replacement rule: '
                      f'argmax(x) --> sum(i * softmax(x[i]))', stacklevel=2)
        debias = 'argmax' in self.debias
        
        def _jax_wrapped_calc_argmax_approx(x, axis, param):
            prob_max = jax.nn.softmax(param * x, axis=axis)
            literals = FuzzyLogic._literals(prob_max.shape, axis=axis)
            sample = jnp.sum(literals * prob_max, axis=axis)
            if debias:
                hard_sample = jnp.argmax(x, axis=axis)
                sample += jax.lax.stop_gradient(hard_sample - sample)
            return sample
        
        tags = ('weight', 'argmax')
        new_param = (tags, self.weight)
        return _jax_wrapped_calc_argmax_approx, new_param
    
    def argmin(self):
        jax_argmax, jax_param = self.argmax()
        
        def _jax_wrapped_calc_argmin_approx(x, axis, param):
            return jax_argmax(-x, axis, param)
        
        return _jax_wrapped_calc_argmin_approx, jax_param
    
    # ===========================================================================
    # control flow
    # ===========================================================================
     
    def If(self):
        warnings.warn('Using the replacement rule: '
                      'if c then a else b --> c * a + (1 - c) * b', stacklevel=2)
        
        def _jax_wrapped_calc_if_approx(c, a, b, param):
            return c * a + (1.0 - c) * b
        
        return _jax_wrapped_calc_if_approx, None
    
    def Switch(self):
        warnings.warn('Using the replacement rule: '
                      'switch(pred) { cases } --> sum(cases[i] * (pred == i))',
                      stacklevel=2)   
        debias = 'Switch' in self.debias
        
        def _jax_wrapped_calc_switch_approx(pred, cases, param):
            pred = jnp.broadcast_to(pred[jnp.newaxis, ...], shape=cases.shape)
            literals = FuzzyLogic._literals(cases.shape, axis=0)
            proximity = -jnp.abs(pred - literals)
            soft_case = jax.nn.softmax(param * proximity, axis=0)
            sample = jnp.sum(cases * soft_case, axis=0)
            if debias:
                hard_case = jnp.argmax(proximity, axis=0)[jnp.newaxis, ...]      
                hard_sample = jnp.take_along_axis(cases, hard_case, axis=0)[0, ...]
                sample += jax.lax.stop_gradient(hard_sample - sample)
            return sample
        
        tags = ('weight', 'switch')
        new_param = (tags, self.weight)
        return _jax_wrapped_calc_switch_approx, new_param
    
    # ===========================================================================
    # random variables
    # ===========================================================================
     
    def _gumbel_softmax(self, key, prob):
        Gumbel01 = random.gumbel(key=key, shape=prob.shape)
        sample = Gumbel01 + jnp.log(prob + self.eps)
        return sample
        
    def bernoulli(self):
        warnings.warn('Using the replacement rule: '
                      'Bernoulli(p) --> Gumbel-softmax(p)', stacklevel=2)
        
        jax_gs = self._gumbel_softmax
        jax_argmax, jax_param = self.argmax()
        
        def _jax_wrapped_calc_switch_approx(key, prob, param):
            prob = jnp.stack([1.0 - prob, prob], axis=-1)
            sample = jax_gs(key, prob)
            sample = jax_argmax(sample, -1, param)
            return sample
        
        return _jax_wrapped_calc_switch_approx, jax_param
    
    def discrete(self):
        warnings.warn('Using the replacement rule: '
                      'Discrete(p) --> Gumbel-softmax(p)', stacklevel=2)
        
        jax_gs = self._gumbel_softmax
        jax_argmax, jax_param = self.argmax()
        
        def _jax_wrapped_calc_discrete_approx(key, prob, param):
            sample = jax_gs(key, prob) 
            sample = jax_argmax(sample, -1, param)
            return sample
        
        return _jax_wrapped_calc_discrete_approx, jax_param
        

# UNIT TESTS
logic = FuzzyLogic()
w = 100.0


def _test_logical():
    print('testing logical')
    _and, _ = logic.And()
    _not, _ = logic.Not()
    _gre, _ = logic.greater()
    _or, _ = logic.Or()
    _if, _ = logic.If()
    
    # https://towardsdatascience.com/emulating-logical-gates-with-a-neural-network-75c229ec4cc9
    def test_logic(x1, x2):
        q1 = _and(_gre(x1, 0, w), _gre(x2, 0, w), w)
        q2 = _and(_not(_gre(x1, 0, w), w), _not(_gre(x2, 0, w), w), w)
        cond = _or(q1, q2, w)
        pred = _if(cond, +1, -1, w)
        return pred
    
    x1 = jnp.asarray([1, 1, -1, -1, 0.1, 15, -0.5]).astype(float)
    x2 = jnp.asarray([1, -1, 1, -1, 10, -30, 6]).astype(float)
    print(test_logic(x1, x2))


def _test_indexing():
    print('testing indexing')
    _argmax, _ = logic.argmax()
    _argmin, _ = logic.argmax()

    def argmaxmin(x):
        amax = _argmax(x, 0, w)
        amin = _argmin(x, 0, w)
        return amax, amin
        
    values = jnp.asarray([2., 3., 5., 4.9, 4., 1., -1., -2.])
    amax, amin = argmaxmin(values)
    print(amax)
    print(amin)


def _test_control():
    print('testing control')
    _switch, _ = logic.Switch()
    
    pred = jnp.asarray(jnp.linspace(0, 2, 10))
    case1 = jnp.asarray([-10.] * 10)
    case2 = jnp.asarray([1.5] * 10)
    case3 = jnp.asarray([10.] * 10)
    cases = jnp.asarray([case1, case2, case3])
    print(_switch(pred, cases, w))


def _test_random():
    print('testing random')
    key = random.PRNGKey(42)
    _bernoulli, _ = logic.bernoulli()
    
    def bern(n):
        prob = jnp.asarray([0.3] * n)
        sample = _bernoulli(key, prob, w)
        return sample
    
    samples = bern(5000)
    print(jnp.mean(samples))


def _test_rounding():
    print('testing rounding')
    _floor, _ = logic.floor()
    _ceil, _ = logic.ceil()
    _mod, _ = logic.mod()
    
    x = jnp.asarray([2.1, 0.5001, 1.99, -2.01, -3.2, -0.1, -1.01, 23.01, -101.99, 200.01])
    print(_floor(x, w))
    print(_ceil(x, w))
    print(_mod(x, 2.0, w))


if __name__ == '__main__':
    _test_logical()
    _test_indexing()
    _test_control()
    _test_random()
    _test_rounding()
    
