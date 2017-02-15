function s = vec2square(v)
% takes a vector of size D^2, reshapes and plots the D x D image.
% check if square, or 3*square (means RGB)

D = numel(v);
sqrtD = sqrt(D);
if mod(sqrtD,1) == 0 % grayscale
    s = reshape(v, round(sqrt(numel(v))) , []);
else % then it's RGB : 
    s = reshape(v, sqrt(D/3), sqrt(D/3), 3);
%     if max(max(max(s)))>1
%         s = s/255; 
%     end
end


end