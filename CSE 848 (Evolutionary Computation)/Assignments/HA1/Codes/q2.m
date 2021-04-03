x0 = [-1, -3];    % initial point
[x, fval, history] = fmincon_driver(x0);
disp(['Min f(x): ', num2str(fval)]);
disp(['x: ', num2str(x)]);

% Plot History
x1 = linspace(-2, 5);    
x2 = linspace(-2, 5);
[X1, X2] = meshgrid(x1, x2);
Z = (X1+1).^2 + (X2+1).^2;

figure;
hold on;
grid on;

title('Search Space');
xlabel('x1');
ylabel('x2');

plot(history(:,1),history(:,2),'r-');
plot(history(:,1),history(:,2),'bo');
contour(X1,X2,Z,'b');
plot(x1,sqrt((5-(x1-1).^2)/4),'g-');
plot(x1,x1-1,'g-');
plot(x1,(2-x1)/2,'g-');


function [x, fval, history] = fmincon_driver(x0) 

    % Objective Function
    f = @(x) ((x(1)+1).^2 + (x(2)+1).^2);

    % Constraint Formation

           % linear
    A = [-1, 1; 1, 2];  
    B = [-1; 2];
    Aeq = []; 
    Beq = []; 
    lb = [];
    ub = [];

            % non-linear
    nonlcon = @nlconstraint;

    history = x0;
    options = optimset('Display', 'Iter', 'PlotFcns', @optimplotfval, 'OutputFcn', @output_check);
    [x, fval] = fmincon(f, x0, A, B, Aeq, Beq, lb, ub, nonlcon, options);
    
    function [stop] = output_check(x, optimValues, state)
        % nesting function to store history
        stop = false;
        if isequal(state, 'iter')
            history = [history; x];
        end
    end
end 


function [c, ceq] = nlconstraint(x)
    % function returning non-linear constraints
    c = (x(1)-1)^2 + 4*x(2)^2 - 5;
    ceq = [];
end