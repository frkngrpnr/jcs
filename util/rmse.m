function [ rmse_perf,mae_perf ] = rmse( u,v )
rmse_perf=(mean((u-v).^2)).^.5;
mae_perf=mean(abs(u-v));
end

