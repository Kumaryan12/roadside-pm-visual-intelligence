# Feature Definitions

## Traffic Load Score

Initial prototype:

traffic_load_score =
1.0 * car_count
+ 0.6 * motorcycle_count
+ 0.9 * auto_rickshaw_count
+ 3.0 * bus_count
+ 3.5 * truck_count
+ 0.1 * bicycle_count

In the research-grade version, these weights should be treated as constrained hyperparameters or replaced by literature-backed emission factors.

## Exhaust Proxy

E_exhaust = sum(N_v * EF_v)

## Resuspension Proxy

E_resuspension = road_dust_score * sum(N_v * RW_v)

## Combined Vehicle PM Proxy

E_vehicle_PM_proxy = alpha * E_exhaust + beta * E_resuspension
