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
using NDTensors
NDTensors.Strided.disable_threads()

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

function compute_two_site_rdm(psi::MPS, i::Int, j::Int)::Matrix{ComplexF64}
    """Compute 2-site reduced density matrix ρ_ij as 4x4 matrix."""
    if i > j
        i, j = j, i
    end
    
    N = length(psi)
    
    # For adjacent sites, use direct contraction
    # For non-adjacent, need more careful handling
    orthogonalize!(psi, i)
    
    # Build the reduced density matrix
    # ρ_ij = Tr_{k≠i,j} |ψ⟩⟨ψ|
    
    # Contract everything to the left of i
    if i > 1
        L_env = psi[1] * dag(prime(psi[1], "Link"))
        for k in 2:(i-1)
            L_env = L_env * psi[k] * dag(prime(psi[k], "Link"))
        end
    else
        L_env = ITensor(1.0)
    end
    
    # Contract everything to the right of j
    if j < N
        R_env = psi[N] * dag(prime(psi[N], "Link"))
        for k in (N-1):-1:(j+1)
            R_env = psi[k] * dag(prime(psi[k], "Link")) * R_env
        end
    else
        R_env = ITensor(1.0)
    end
    
    # Contract sites between i and j (excluding i and j)
    M_env = ITensor(1.0)
    for k in (i+1):(j-1)
        M_env = M_env * psi[k] * dag(prime(psi[k], "Link"))
    end
    
    # Build ρ_ij
    si = siteind(psi, i)
    sj = siteind(psi, j)
    
    # Contract: L_env * psi[i] * M_env * psi[j] * R_env * dag(psi[i]') * dag(psi[j]')
    rho_tensor = L_env * psi[i]
    if i + 1 < j
        rho_tensor = rho_tensor * M_env
    end
    rho_tensor = rho_tensor * psi[j] * R_env
    rho_tensor = rho_tensor * dag(prime(psi[i], si))
    if i + 1 < j
        # Need to handle intermediate links
        for k in (i+1):(j-1)
            rho_tensor = rho_tensor * dag(prime(psi[k], "Link"))
        end
    end
    rho_tensor = rho_tensor * dag(prime(psi[j], sj))
    
    # Convert to 4x4 matrix
    # Basis: |00⟩, |01⟩, |10⟩, |11⟩ (0=down, 1=up)
    rdm = zeros(ComplexF64, 4, 4)
    
    for a in 1:2, b in 1:2, c in 1:2, d in 1:2
        row = (a-1) * 2 + b
        col = (c-1) * 2 + d
        rdm[row, col] = rho_tensor[si => a, si' => c, sj => b, sj' => d]
    end
    
    return rdm
end

function compute_single_site_rdm(psi::MPS, i::Int)::Matrix{ComplexF64}
    """Compute single-site RDM as 2x2 matrix."""
    orthogonalize!(psi, i)
    
    si = siteind(psi, i)
    rho_tensor = psi[i] * dag(prime(psi[i], si))
    
    rdm = zeros(ComplexF64, 2, 2)
    for a in 1:2, b in 1:2
        rdm[a, b] = rho_tensor[si => a, si' => b]
    end
    
    return rdm
end

function compute_nn_rdms(psi::MPS, L::Int)::Vector{Matrix{ComplexF64}}
    """Compute all nearest-neighbor 2-site RDMs."""
    N = L * L
    rdms = Matrix{ComplexF64}[]
    
    for iy in 1:L
        for ix in 1:L
            i = site_to_index(ix, iy, L)
            
            # Right neighbor
            jx = mod1(ix + 1, L)
            j = site_to_index(jx, iy, L)
            if j > i  # Avoid duplicates
                push!(rdms, compute_two_site_rdm(psi, i, j))
            end
            
            # Up neighbor
            jy = mod1(iy + 1, L)
            j = site_to_index(ix, jy, L)
            if j > i
                push!(rdms, compute_two_site_rdm(psi, i, j))
            end
        end
    end
    
    return rdms
end

function rdms_to_vector(rdms::Vector{Matrix{ComplexF64}})::Vector{Float64}
    """Flatten RDMs to real vector for VAE input."""
    # Each 4x4 complex RDM -> 32 real numbers (16 real + 16 imag)
    vec = Float64[]
    for rdm in rdms
        for val in rdm
            push!(vec, real(val))
            push!(vec, imag(val))
        end
    end
    return vec
end

function compute_observables(psi::MPS, sites, L::Int, energy::Float64)::Vector{Float64}
    N = L * L
    observables = zeros(Float64, 11)
    
    observables[1] = energy
    observables[2] = energy / N
    
    # Staggered magnetization
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
    
    # Structure factors
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
    
    return observables
end

function run_dmrg_rdm(L::Int, j2_j1::Float64, bond_dim::Int, output_file::String)
    N = L * L
    
    println("=" ^ 60)
    println("J1-J2 DMRG with RDM extraction")
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
    
    println("Computing observables...")
    flush(stdout)
    observables = compute_observables(psi, sites, L, energy)
    
    println("Computing single-site RDMs...")
    flush(stdout)
    single_rdms = [compute_single_site_rdm(psi, i) for i in 1:N]
    single_rdm_vec = Float64[]
    for rdm in single_rdms
        for val in rdm
            push!(single_rdm_vec, real(val))
            push!(single_rdm_vec, imag(val))
        end
    end
    
    println("Computing NN 2-site RDMs...")
    flush(stdout)
    # For simplicity, compute RDMs for a subset of bonds (all would be expensive)
    # Use bonds along a row and column through center
    nn_rdms = Matrix{ComplexF64}[]
    mid = L ÷ 2 + 1
    
    # Horizontal bonds in middle row
    for ix in 1:L
        i = site_to_index(ix, mid, L)
        jx = mod1(ix + 1, L)
        j = site_to_index(jx, mid, L)
        if abs(i - j) == 1  # Only adjacent in MPS ordering
            push!(nn_rdms, compute_two_site_rdm(psi, min(i,j), max(i,j)))
        end
    end
    
    # Vertical bonds in middle column
    for iy in 1:L
        i = site_to_index(mid, iy, L)
        jy = mod1(iy + 1, L)
        j = site_to_index(mid, jy, L)
        if abs(i - j) == L  # Vertical neighbors differ by L in row-major
            push!(nn_rdms, compute_two_site_rdm(psi, min(i,j), max(i,j)))
        end
    end
    
    nn_rdm_vec = rdms_to_vector(nn_rdms)
    
    # Combine into single feature vector
    rdm_features = vcat(single_rdm_vec, nn_rdm_vec)
    
    println("Saving to $output_file...")
    flush(stdout)
    
    h5open(output_file, "w") do f
        f["energy"] = energy
        f["observables"] = observables
        f["rdm_features"] = rdm_features
        f["single_rdm_vec"] = single_rdm_vec
        f["L"] = L
        f["j2_j1"] = j2_j1
        f["bond_dim"] = maxlinkdim(psi)
        f["n_single_rdms"] = N
        f["n_nn_rdms"] = length(nn_rdms)
        f["rdm_feature_dim"] = length(rdm_features)
        
        obs_names = ["energy", "energy_density", "staggered_magnetization",
                     "stripe_order", "plaquette_order", "S_pi_pi", "S_pi_0",
                     "entanglement_entropy", "nematic_order", "dimer_order_x", "dimer_order_y"]
        f["observable_names"] = obs_names
    end
    
    println("Done! RDM feature dim = $(length(rdm_features))")
    println("=" ^ 60)
    
    return energy, observables, rdm_features
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
