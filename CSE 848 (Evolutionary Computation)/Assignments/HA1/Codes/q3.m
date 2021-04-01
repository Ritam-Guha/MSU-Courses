% Setting up Coefficients of Obj Function and Constraints
C = [1, 4];
A = [1, 5; 3, 1; -1, -2];
B = [10; 15; -1];

% Linear Programming approach
[x_lin, fval] = linprog(-C, A, B);  
x_lin = x_lin';

disp(['Max f(x): ', num2str(-fval)]);
disp(['x: ', num2str(x_lin)]);

% Restricting first variable to integer values for Integer Programming 
intcon = [1];
[x_intlin, fval] = intlinprog(-C, intcon, A, B);  
x_intlin = x_intlin';

disp(['Max f(x): ', num2str(-fval)]);
disp(['x: ', num2str(x_intlin)]);

% Plotting feasible regions 
x1 = linspace(0, 5);    
x2 = linspace(0, 5);
[X1, X2] = meshgrid(x1, x2);
Z = X1 + 4*X2;

figure;
hold on;
grid on;

title('Search Space')
xlabel('x1')
ylabel('x2')

contour(X1, X2, Z, 'r');
plot(x1, (10-x1)/5, 'g-');  
text(0,2,'x_1+5x_2=10')
text(1,5,'z=x_1+4x_2','Color','r')
plot(x1, 15-3*x1, 'g-'); 
text(3,6,'3x_1+x_2=15')
plot(x1, (1-x1)/2, 'g-');
text(3,-1,'x_1+2x_2=1')
plot(x_lin(1), x_lin(2), 'r*');
text(x_lin(1), x_lin(2), 'lin');
plot(x_intlin(1), x_intlin(2), 'b*');
text(x_intlin(1), x_intlin(2), 'intlin');
hold off;
