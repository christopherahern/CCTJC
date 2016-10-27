from scipy.stats import beta, uniform
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
hfont = {'fontname':'Times New Roman'}



x = np.arange (0, 1, 0.01)
t = .2
b = .25
U_S = [-(a - t - b)**2 for a in x]
U_R = [-(a - t)**2 for a in x]
plt.plot(x, U_S, label=r'$U_S$')
plt.plot(x, U_R, label=r'$U_R$')
plt.ylim(-1.0, .05)
plt.legend(loc='lower right')
plt.ylabel('Utility', fontsize=15, **hfont)
plt.xlabel("Action", fontsize=15, **hfont)
plt.show()


x = np.arange (0, 1, 0.01)
t = .2
b = .25
U_S = [1 - (a - t - (1-t)*b)**2 for a in x]
U_R = [1 - (a - t)**2 for a in x]
plt.plot(x, U_S, label=r'$U_S$')
plt.plot(x, U_R, label=r'$U_R$')
plt.ylim(0, 1.05)
plt.legend(loc='lower right')
plt.ylabel('Utility', fontsize=15, **hfont)
plt.xlabel("Action", fontsize=15, **hfont)
plt.show()


x = np.arange (0, 1, 0.01)
t = .2
for b in np.linspace(0, .5, num=3):
    U_S = [1 -(a - t - b)**2 for a in x]
    plt.plot(x, U_S, label=r'$b=$ ' + str(b))
plt.ylim(0, 1.05)
plt.legend(loc='lower right')
plt.ylabel('Utility', fontsize=15, **hfont)
plt.xlabel("Action", fontsize=15, **hfont)
plt.show()


x = np.arange (0, 1, 0.01)
b = .3
plt.plot(x, x + (1-x)*b)
plt.ylim(0, 1.05)
plt.plot(x, x, 'k--')
plt.legend(loc='lower right')
plt.ylabel('Preferred action', fontsize=15, **hfont)
plt.xlabel("State", fontsize=15, **hfont)
plt.show()


x = np.arange (0, 1, 0.01)
b = .3
y = -x**3 + x**2 + x
plt.plot(x, y)
plt.ylim(0, 1.05)
plt.plot(x, x, 'k--')
plt.legend(loc='lower right')
plt.ylabel('Preferred action', fontsize=15, **hfont)
plt.xlabel("State", fontsize=15, **hfont)
plt.show()



x = np.arange (0, 1, 0.01)

plt.style.use('ggplot')
for beta_var in range(1,6):
        y = beta.pdf(x,1,beta_var)
        plt.plot(x,y, label=r'$\beta_p = $' + str(beta_var), linewidth=2)
plt.legend(loc='upper right')
plt.ylabel("Probability", fontsize=15, **hfont)
plt.xlabel("State", fontsize=15, **hfont)
plt.savefig('../local/out/beta-distribution.pdf', format='pdf', dpi=1000)
plt.show()
