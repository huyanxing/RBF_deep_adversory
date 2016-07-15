function grad = sigmoidGrad(x)
    e_x = exp(-x);
    grad = e_x ./ ((1 + e_x).^2); 
end

