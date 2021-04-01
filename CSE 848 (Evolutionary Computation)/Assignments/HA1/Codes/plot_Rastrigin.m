[x1,x2]=meshgrid(-5.12:0.01:5.12,-5.12:0.01:5.12);
f=rastrigin([x1(:) x2(:)]);
disp(size(f))
disp(size(x1))
f=reshape(f,size(x1));
surf(x1,x2,f,'linestyle','none');axis tight;


function [func_val] = rastrigin(x)
    % Formulation of n-dimensional Rastrigin function
    alpha = 2.0;   % set the value of alpha 
    n = size(x, 2);
    func_val = 10*n + sum(x.^2 - 10*cos(alpha * pi * x), 2); 
end
