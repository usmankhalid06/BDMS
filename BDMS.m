function [sources, mixing_matrix, convergence_errors, C] = BDMS(Y, num_sources, lambda, max_iter,zeta1,zeta2,zeta3,zeta4, Dp, zeta5, TC, SM)

[n_timepoints, n_voxels] = size(Y);
[U, S, Z] = svds(Y, num_sources);
W = U';
% rng(5, 'twister'); 
% W = rand(num_sources, size(Y,1));  W = W' * diag(1./sqrt(sum(W'.*W')));    W = W';
learning_rate = 0.01;
convergence_errors = zeros(max_iter, 1);

S = W * Y;
fprintf('Iteration:     ');
for iter = 1:max_iter
    fprintf('\b\b\b\b\b%5i',iter);

    Wo = W;

    F1 = W*W'; G1 = W*Y;
    for i =1:num_sources
        sk = 1.0/F1(i,i) * (G1(i,:) - F1(i,:)*S) + S(i,:);
        thr = lambda./(abs(sk));
        S(i,:) = sign(sk).*max(0, bsxfun(@minus,abs(sk),thr/2));
    end

    dW = zeros(size(W));
    for i = 1:num_sources
        % Kurtosis gradient (4th moment) on thresholded sources
        s_centered = S(i,:) - mean(S(i,:));
        grad_kurt = 4 * (s_centered.^3) * Y' / n_voxels;
        
        % Spectral entropy gradient on thresholded sources
        S_fft = fft(S(i,:));
        power_spectrum = abs(S_fft).^2;
        grad_spectral = -real(ifft(log(power_spectrum + eps) .* S_fft)) * Y' / n_voxels;

        % Hoyer sparsity gradient on thresholded sources
        l1_norm = sum(abs(S(i,:)));
        l2_norm = sqrt(sum(S(i,:).^2));
        grad_hoyer = (sign(S(i,:))/(l2_norm + eps) - l1_norm*S(i,:)/(l2_norm^3 + eps)) * Y' / n_voxels;

        % Reconstruction error
        reconstruction_error = W(i,:)*Y - S(i,:);  % Only for source i
        grad_recon = -reconstruction_error * Y' / n_voxels;

        % Magnitude matching
        spatial_scale = zeta1* norm(grad_kurt) / (norm(grad_spectral) + eps); %0.1 0.3
        hoyer_scale = zeta2 * norm(grad_kurt) / (norm(grad_hoyer) + eps);
        recon_scale = zeta3 * norm(grad_kurt) / (norm(grad_recon) + eps); %0.05

        % Combine gradients
        dW(i,:) = zeta4 * grad_kurt + spatial_scale * grad_spectral + hoyer_scale * grad_hoyer + recon_scale * grad_recon; 

%         sigma_time = 0.2* n_timepoints;     % ~10% of signal length
%         sigma_intensity = 0.5* std(dW(i,:)');    % 20% of signal standard deviation
%         W2 = bilateral_temporal_filter(dW(i,:)', sigma_time, sigma_intensity);
%         dW(i,:) =W2';

    end

    W = W + learning_rate * dW;
    W = W' * diag(1./sqrt(sum(W'.*W')));
    W = W';

    convergence_errors(iter) = sqrt(trace((W - Wo)' * (W - Wo))) / (sqrt(trace(Wo' * Wo))+ eps);
  
    K2 = size(TC,2);
    [~,~,ind]=sort_TSandSM_spatial(TC,SM,W',S,K2);
    for ii =1:K2
        TCcorr(ii) =abs(corr(TC(:,ii),W(ind(ii),:)'));
        SMcorr(ii) =abs(corr(SM(ii,:)',S(ind(ii),:)'));
    end
    cTC = sum(TCcorr');
    cSM = sum(SMcorr');
    C(iter) =cTC+cSM;


end

sources = S;
mixing_matrix = W';

% 
% for k = 1:num_sources
%     sorted_vals = sort(abs(mixing_matrix(:, k)));
%     threshold = sorted_vals(round(0.9 * length(sorted_vals)));
%     mixing_matrix(:, k) = sign(mixing_matrix(:, k)) .* min(abs(mixing_matrix(:, k)), threshold);
% end


A = zeros(size(Dp,2),num_sources);
for k = 1:num_sources
    [~,bb]= sort(abs(Dp'*mixing_matrix(:,k)),'descend');
    ind = bb(1:zeta5);
    A(ind,k)= (Dp(:,ind)'*Dp(:,ind))\Dp(:,ind)'*mixing_matrix(:,k);
    A(:,k) = A(:,k)./norm(Dp*A(:,k));
    mixing_matrix(:,k) = Dp*A(:,k);
end



end
% mixing_matrix = sgolayfilt(mixing_matrix, 3, 7);
