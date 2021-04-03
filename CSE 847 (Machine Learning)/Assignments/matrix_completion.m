clear all;

B = randn(4,6);
A = randn(7,4);
C_orig = A * B;  
C(:,:,1) = C_orig;

rank(C)
pause

num_missing = 11;
missing_pos = randi(5, num_missing, 2);

for i=1:num_missing
    C(missing_pos(i,1), missing_pos(i,2),1)=0;
end

fprintf('Initial Norm: %f\n', norm(C(:,:,1)-C_orig, 'fro'));

pause

idx=2;
r=2;

while true
    [U, S, V] = svd(C(:,:,idx-1));
    C(:,:,idx) = U(:, 1:r) * S(1:r, 1:r) * V(:, 1:r)';
    if(norm(C(:,:,idx)-C(:,:,idx-1), 'fro')<0.01)
        break
    end
    idx = idx+1;
end
fprintf('Iterations-%d Final Norm: %f\n', idx, norm(C(:,:,idx)-C_orig, 'fro'));


    
