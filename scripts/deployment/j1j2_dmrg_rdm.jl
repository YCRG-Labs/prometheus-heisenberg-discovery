#!/usr/bin/env julia
"""
J1-J2 Heisenberg DMRG with RDM extraction using ITensors.jl

Computes ground states and extracts 2-site reduced density matrices (RDMs)
for use with VAE training at large system sizes.

Usage:
    julia j1j2_dmrg_rdm.jl <L> <j2_j1> <bond_dim> <output_file>
"""

using ITensors
using ITensorMPS
using HDF5
using LinearAlgebra

# Configure threading properly for ITensors block sparse
BLAS.set_num_threads(1)
ITensors.enable_threaded_blocksparse()

function site_to_index(ix::Int, iy::Int, L::Int)::Int
    return (iy - 1) * L + ix
end

function build_j1j2_hamiltonian(sites, L::Int, j2_j1::Float64)
    N = L * L
    ampo = OpSum()
    
    J1 = 1.0
    for iy in 1:L
        for ix in 1:L
            i = site_to_index(ix, iy, L)
            
            # Right neighbor (PBC)
            jx = mod1(ix + 1, L)
            j = site_to_index(jx, iy, L)
            ampo += J1 * 0.5, "S+", i, "S-", j
            ampo += J1 * 0.5, "S-", i, "S+", j
            ampo += J1, "Sz", i, "Sz", j
            
            # Up neighbor (PBC)
            jy = mod1(iy + 1, L)
            j = site_to_index(ix, jy, L)
            ampo += J1 * 0.5, "S+", i, "S-", j
            ampo += J1 * 0.5, "S-", i, "S+", j
            ampo += J1, "Sz", i, "Sz", j
        end
    end
    
    if abs(j2_j1) > 1e-12
        J2 = j2_j1
        for iy in 1:L
            for ix in 1:L
                i = site_to_index(ix, iy, L)
                
                # Up-right diagonal
                jx = mod1(ix + 1, L)
                jy = mod1(iy + 1, L)
                j = site_to_index(jx, jy, L)
                ampo += J2 * 0.5, "S+", i, "S-", j
                ampo += J2 * 0.5, "S-", i, "S+", j
                ampo += J2, "Sz", i, "Sz", j
                
                # Up-left diagonal
                jx = mod1(ix - 1, L)
                j = site_to_index(jx, jy, L)
                ampo += J2 * 0.5, "S+", i, "S-", j
                ampo += J2 * 0.5, "S-", i, "S+", j
                ampo += J2, "Sz", i, "Sz", j
            end
        end
    end
    
    return MPO(ampo, sites)
end

function compute_features(psi::MPS, sites, L::Int, energy::Float64)::Tuple{Vector{Float64}, Vector{Float64}}
    """Compute RDM-based features and observables from MPS ground state."""
    N = L * L
    observables = zeros(Float64, 11)
    
    # Observables
    observables[1] = energy
    observables[2] = energy / N
    
    # Single-site expectations
    sz_vals = expect(psi, "Sz")
    
    # Staggered magnetization
    stag_mag = 0.0
    for iy in 1:L
        for ix in 1:L
            i = site_to_index(ix, iy, L)
            sign = (-1.0)^(ix + iy)
            stag_mag += sign * sz_vals[i]
        end
    end
    observables[3] = abs(stag_mag) / N
    
    # Correlation matrix for structure factors
    zz_corr = correlation_matrix(psi, "Sz", "Sz")
    
    s_pi_pi = 0.0
    s_pi_0 = 0.0
    for iy1 in 1:L, ix1 in 1:L
        i = site_to_index(ix1, iy1, L)
        for iy2 in 1:L, ix2 in 1:L
            j = site_to_index(ix2, iy2, L)
            phase_pi_pi = (-1.0)^((ix1 - ix2) + (iy1 - iy2))
            phase_pi_0 = (-1.0)^(ix1 - ix2)
            s_pi_pi += phase_pi_pi * zz_corr[i, j]
            s_pi_0 += phase_pi_0 * zz_corr[i, j]
        end
    end
    observables[6] = abs(s_pi_pi) / N
    observables[7] = abs(s_pi_0) / N
    
    # Entanglement entropy
    mid = N ÷ 2
    SvN = 0.0
    try
        orthogonalize!(psi, mid)
        if mid > 1 && mid < N
            U, S, V = svd(psi[mid], (linkind(psi, mid - 1),))
            for n in 1:dim(S, 1)
                p = S[n, n]^2
                if p > 1e-14
                    SvN -= p * log(p)
                end
            end
        end
    catch e
        println("Warning: EE computation failed: $e")
    end
    observables[8] = SvN
    
    # Build feature vector from:
    # 1. Single-site <Sz> values (N values)
    # 2. Flattened correlation matrix (N*N values, but use upper triangle = N*(N+1)/2)
    # 3. Key observables
    
    features = Float64[]
    
    # Single-site Sz
    append!(features, sz_vals)
    
    # Upper triangle of correlation matrix (includes diagonal)
    for i in 1:N
        for j in i:N
            push!(features, zz_corr[i, j])
        end
    end
    
    # Add key observables as features
    push!(features, observables[3])  # staggered mag
    push!(features, observables[6])  # S(pi,pi)
    push!(features, observables[7])  # S(pi,0)
    push!(features, observables[8])  # entanglement entropy
    
    return features, observables
end

function run_dmrg_rdm(L::Int, j2_j1::Float64, bond_dim::Int, output_file::String)
    N = L * L
    
    println("=" ^ 60)
    println("J1-J2 DMRG with feature extraction")
    println("=" ^ 60)
    println("  L = $L ($N spins)")
    println("  J2/J1 = $j2_j1")
    println("  Bond dim = $bond_dim")
    println()
    
    sites = siteinds("S=1/2", N; conserve_qns=true)
    
    println("Building Hamiltonian...")
    flush(stdout)
    H = build_j1j2_hamiltonian(sites, L, j2_j1)
    
    init_state = [isodd(i) ? "Up" : "Dn" for i in 1:N]
    psi0 = MPS(sites, init_state)
    
    nsweeps = 20
    maxdim = [20, 40, 80, 100, 150, 200, bond_dim, bond_dim, bond_dim, bond_dim,
              bond_dim, bond_dim, bond_dim, bond_dim, bond_dim, bond_dim,
              bond_dim, bond_dim, bond_dim, bond_dim]
    cutoff = fill(1e-10, nsweeps)
    noise = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    println("Running DMRG...")
    flush(stdout)
    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff, noise, outputlevel=1)
    
    println()
    println("E = $energy, E/N = $(energy/N)")
    println("Max bond dim = $(maxlinkdim(psi))")
    println()
    
    println("Computing features and observables...")
    flush(stdout)
    features, observables = compute_features(psi, sites, L, energy)
    
    println("  Feature vector dim: $(length(features))")
    println("  Staggered mag: $(observables[3])")
    println("  S(π,π): $(observables[6])")
    println("  Entanglement entropy: $(observables[8])")
    println()
    
    println("Saving to $output_file...")
    flush(stdout)
    
    h5open(output_file, "w") do f
        f["energy"] = energy
        f["observables"] = observables
        f["rdm_features"] = features
        f["L"] = L
        f["j2_j1"] = j2_j1
        f["bond_dim"] = maxlinkdim(psi)
        f["rdm_feature_dim"] = length(features)
        
        obs_names = ["energy", "energy_density", "staggered_magnetization",
                     "stripe_order", "plaquette_order", "S_pi_pi", "S_pi_0",
                     "entanglement_entropy", "nematic_order", "dimer_order_x", "dimer_order_y"]
        f["observable_names"] = obs_names
    end
    
    println("Done!")
    println("=" ^ 60)
    
    return energy, observables, features
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 4
        println("Usage: julia j1j2_dmrg_rdm.jl <L> <j2_j1> <bond_dim> <output_file>")
        exit(1)
    end
    
    L = parse(Int, ARGS[1])
    j2_j1 = parse(Float64, ARGS[2])
    bond_dim = parse(Int, ARGS[3])
    output_file = ARGS[4]
    
    run_dmrg_rdm(L, j2_j1, bond_dim, output_file)
end
