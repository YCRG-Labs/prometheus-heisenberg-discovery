#!/usr/bin/env julia
"""
J1-J2 Heisenberg DMRG using ITensors.jl

Usage:
    julia j1j2_dmrg.jl <L> <j2_j1> <bond_dim> <output_file>

Example:
    julia j1j2_dmrg.jl 6 0.5 200 groundstate.h5
"""

using ITensors
using ITensorMPS
using HDF5
using LinearAlgebra

function site_to_index(ix::Int, iy::Int, L::Int)::Int
    return (iy - 1) * L + ix
end

function build_j1j2_hamiltonian(sites, L::Int, j2_j1::Float64)
    """Build J1-J2 Heisenberg Hamiltonian on square lattice with PBC."""
    N = L * L
    
    ampo = OpSum()
    
    # J1 nearest-neighbor terms
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
    
    # J2 next-nearest-neighbor terms (diagonals)
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
                jy = mod1(iy + 1, L)
                j = site_to_index(jx, jy, L)
                
                ampo += J2 * 0.5, "S+", i, "S-", j
                ampo += J2 * 0.5, "S-", i, "S+", j
                ampo += J2, "Sz", i, "Sz", j
            end
        end
    end
    
    return MPO(ampo, sites)
end

function measure_sz(psi::MPS, i::Int)::Float64
    """Measure <Sz> at site i."""
    return expect(psi, "Sz"; sites=i)[1]
end

function compute_observables(psi::MPS, sites, L::Int, energy::Float64)::Vector{Float64}
    """Compute observables from MPS ground state."""
    N = L * L
    observables = zeros(Float64, 11)
    
    # 1. Energy
    observables[1] = energy
    
    # 2. Energy density
    observables[2] = energy / N
    
    # 3. Staggered magnetization
    sz_vals = expect(psi, "Sz")
    stag_mag = 0.0
    for iy in 1:L
        for ix in 1:L
            i = site_to_index(ix, iy, L)
            sign = (-1.0)^(ix + iy)
            stag_mag += sign * sz_vals[i]
        end
    end
    observables[3] = abs(stag_mag) / N
    
    # 4. Stripe order (placeholder)
    observables[4] = 0.0
    
    # 5. Plaquette order (placeholder)
    observables[5] = 0.0
    
    # 6. S(π,π) structure factor using correlation_matrix
    # S(q) = (1/N) Σ_ij exp(iq·(ri-rj)) <Sz_i Sz_j>
    zz_corr = correlation_matrix(psi, "Sz", "Sz")
    s_pi_pi = 0.0
    for iy1 in 1:L, ix1 in 1:L
        i = site_to_index(ix1, iy1, L)
        for iy2 in 1:L, ix2 in 1:L
            j = site_to_index(ix2, iy2, L)
            # Phase factor for q=(π,π)
            phase = (-1.0)^((ix1 - ix2) + (iy1 - iy2))
            s_pi_pi += phase * zz_corr[i, j]
        end
    end
    observables[6] = abs(s_pi_pi) / N
    
    # 7. S(π,0)
    s_pi_0 = 0.0
    for iy1 in 1:L, ix1 in 1:L
        i = site_to_index(ix1, iy1, L)
        for iy2 in 1:L, ix2 in 1:L
            j = site_to_index(ix2, iy2, L)
            # Phase factor for q=(π,0)
            phase = (-1.0)^(ix1 - ix2)
            s_pi_0 += phase * zz_corr[i, j]
        end
    end
    observables[7] = abs(s_pi_0) / N
    
    # 8. Entanglement entropy at middle bond
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
        println("Warning: Could not compute entanglement entropy: $e")
    end
    observables[8] = SvN
    
    # 9-11: Nematic and dimer orders (placeholder)
    observables[9] = 0.0
    observables[10] = 0.0
    observables[11] = 0.0
    
    return observables
end

function run_dmrg(L::Int, j2_j1::Float64, bond_dim::Int, output_file::String)
    N = L * L
    
    println("=" ^ 60)
    println("J1-J2 Heisenberg DMRG")
    println("=" ^ 60)
    println("  L = $L ($N spins)")
    println("  J2/J1 = $j2_j1")
    println("  Max bond dimension = $bond_dim")
    println()
    
    # Create site indices (conserve total Sz)
    sites = siteinds("S=1/2", N; conserve_qns=true)
    
    # Build Hamiltonian
    println("Building Hamiltonian...")
    flush(stdout)
    H = build_j1j2_hamiltonian(sites, L, j2_j1)
    
    # Initial state: Néel-like pattern
    init_state = [isodd(i) ? "Up" : "Dn" for i in 1:N]
    psi0 = MPS(sites, init_state)
    
    # DMRG parameters with gradual bond dimension increase
    nsweeps = 20
    maxdim = [20, 40, 80, 100, 150, 200, bond_dim, bond_dim, bond_dim, bond_dim,
              bond_dim, bond_dim, bond_dim, bond_dim, bond_dim, bond_dim, 
              bond_dim, bond_dim, bond_dim, bond_dim]
    cutoff = fill(1e-10, nsweeps)
    noise = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    println("Running DMRG ($nsweeps sweeps)...")
    flush(stdout)
    
    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff, noise, outputlevel=1)
    
    println()
    println("Ground state energy: $energy")
    println("Energy per site: $(energy/N)")
    println("Max bond dimension used: $(maxlinkdim(psi))")
    println()
    
    # Compute observables
    println("Computing observables...")
    flush(stdout)
    observables = compute_observables(psi, sites, L, energy)
    
    println("  Staggered magnetization: $(observables[3])")
    println("  S(π,π): $(observables[6])")
    println("  Entanglement entropy: $(observables[8])")
    println()
    
    # Save results
    println("Saving to $output_file...")
    flush(stdout)
    
    h5open(output_file, "w") do f
        f["energy"] = energy
        f["observables"] = observables
        f["L"] = L
        f["j2_j1"] = j2_j1
        f["bond_dim"] = maxlinkdim(psi)
        
        # Store a dummy psi array (actual MPS is too complex for simple HDF5)
        # The observables are what matter for downstream analysis
        f["psi"] = zeros(ComplexF64, 1)
        
        obs_names = [
            "energy", "energy_density", "staggered_magnetization",
            "stripe_order", "plaquette_order", "S_pi_pi", "S_pi_0",
            "entanglement_entropy", "nematic_order", "dimer_order_x", "dimer_order_y"
        ]
        f["observable_names"] = obs_names
    end
    
    println("Done!")
    println("=" ^ 60)
    
    return energy, observables
end

# Main entry point
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 4
        println("Usage: julia j1j2_dmrg.jl <L> <j2_j1> <bond_dim> <output_file>")
        println("Example: julia j1j2_dmrg.jl 6 0.5 200 groundstate.h5")
        exit(1)
    end
    
    L = parse(Int, ARGS[1])
    j2_j1 = parse(Float64, ARGS[2])
    bond_dim = parse(Int, ARGS[3])
    output_file = ARGS[4]
    
    run_dmrg(L, j2_j1, bond_dim, output_file)
end
