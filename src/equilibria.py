from sympy import *
from sympy.stats import Beta, density


def main():
    t, t_0, m_0, m_1, a_0, a_1, b = symbols('t t_0 m_0 m_1 a_0 a_1 b')
    T = Beta("t", 1, 2)
    Utility_S_0 = 1 - (a_0 - t - (1-t)*b)**2
    Utility_S_1 = 1 - (a_1 - t - (1-t)*b)**2

    E_Utility_S = integrate(Utility_S_0*density(T)(t).evalf(), (t, 0, t_0)) + \
                    integrate(Utility_S_1*density(T)(t).evalf(), (t, t_0, 1))
    E_Utility_R = E_Utility_S.subs(b, 0)

    t0_sol = Eq(solve(diff(E_Utility_S, t_0), t_0)[0], t_0)
    a0_sol = Eq(solve(diff(E_Utility_R, a_0), a_0)[0], a_0)
    a1_sol = Eq(solve(diff(E_Utility_R, a_1), a_1)[0], a_1)

    ESS = solve([t0_sol, a0_sol, a1_sol], [t_0, a_0, a_1])[1]

    solve(ESS[0], b)

    x = np.linspace(0,1/6.0, num=100)

    plt.style.use('ggplot')
    plt.plot(x, [ESS[1].subs(b, value).evalf() for value in x], 'r', linewidth=4, linestyle='--')
    plt.plot(x, [ESS[2].subs(b, value).evalf() for value in x], 'b', linewidth=4, linestyle='--')
    plt.plot(x, [ESS[0].subs(b, value).evalf() for value in x], 'k', linewidth=4)
    plt.axhline(1/3.0, 1/6.0, 1, color='b', linewidth=4, ls='--')
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.xlabel(r"$b$", fontsize=18, **hfont)
    plt.ylabel(r"Actions and States", fontsize=18, **hfont)
    # plt.savefig("../local/out/ESS-beta.pdf", format='pdf', dpi=1000, fontsize=18)
    plt.show()

if __name__=="__main__":
    main()
