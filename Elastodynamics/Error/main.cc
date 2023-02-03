/* ---------------------------------------------------------------------
 * The space time finite elements code has been based on Step-3
 * of the deal.II tutorial programs.
 *
 * STEP-3:
 * =======
 *
 * Copyright (C) 1999 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 *
 * Authors: Wolfgang Bangerth, 1999,
 *          Guido Kanschat, 2011
 *
 * ---------------------------------------------------------------------
 * TENSOR-PRODUCT SPACE-TIME FINITE ELEMENTS W/ TIME-SLABBING:
 * ===========================================================
 * Tensor-product space-time code for the elastodynamics equation with Q^s finite elements in space and Q^r finite elements in time: cG(s)cG(r)
 * The time interval I = (0,T) is being divided into subintervals I_1, I_2, ..., I_M 
 * and then instead of solving the PDE all at once on the full space-time domain Q := I x Ω
 * we solve sequentially (forward-in-time) on time slabs Q_n := I_n x Ω to reduce the size of the linear systems.
 * This also allows the usage of different temporal polynomial degrees on each slab, i.e. r := (r_1, r_2, ..., r_M) is now a multiindex.
 * 
 * Author: Hendrik Fischer and Julian Roth, 2022-2023
 * 
 * For more information on tensor-product space-time finite elements please also check out the DTM-project by Uwe Köcher and contributors.
 */

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/convergence_table.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <fstream>
#include <iostream>
#include <cmath>
#include <numeric>
#include <string>
#include <set>
#include <memory>
#include <sys/stat.h> // for mkdir

using namespace dealii;

void print_as_numpy_arrays_high_resolution(SparseMatrix<double> &matrix,
					    std::ostream &     out,
                                            const unsigned int precision)
{
  AssertThrow(out.fail() == false, ExcIO());

  out.precision(precision);
  out.setf(std::ios::scientific, std::ios::floatfield);

  std::vector<int> rows;
  std::vector<int> columns;
  std::vector<double> values;
  rows.reserve(matrix.n_nonzero_elements());
  columns.reserve(matrix.n_nonzero_elements());
  values.reserve(matrix.n_nonzero_elements());

  SparseMatrixIterators::Iterator< double, false > it = matrix.begin();
  for (unsigned int i = 0; i < matrix.m(); i++) {
 	 for (it = matrix.begin(i); it != matrix.end(i); ++it) {
 		rows.push_back(i);
 		columns.push_back(it->column());
		values.push_back(matrix.el(i,it->column()));
 	 }
  }

  for (auto d : values)
    out << d << ' ';
  out << '\n';

  for (auto r : rows)
    out << r << ' ';
  out << '\n';

  for (auto c : columns)
    out << c << ' ';
  out << '\n';
  out << std::flush;

  AssertThrow(out.fail() == false, ExcIO());
}

template<int dim>
class InitialValues: public Function<dim> {
public:
	InitialValues() : Function<dim>(2) {}

	virtual double value(const Point<dim> &p,
			const unsigned int component) const override;

	virtual void vector_value (const Point<dim> &p, 
			     Vector<double>   &value) const;
};

template<int dim>
double InitialValues<dim>::value(const Point<dim> &p,
		const unsigned int component) const {
	if (component == 0)
	  return 0.;
	else if (component == 1)
	  return p[0] * (1. - p[0]);
}

template <int dim>
void InitialValues<dim>::vector_value (const Point<dim> &p,
				    Vector<double>   &values) const 
{
  for (unsigned int c=0; c<2; ++c)
    values(c) = InitialValues<dim>::value(p, c);
}

template<int dim>
class InitialValues2D: public Function<dim> {
public:
	InitialValues2D() : Function<dim>(2) {}

	virtual double value(const Point<dim> &p,
			const unsigned int component) const override;

	virtual void vector_value (const Point<dim> &p, 
			     Vector<double>   &value) const;
};

template<int dim>
double InitialValues2D<dim>::value(const Point<dim> &p,
		const unsigned int component) const {
	double x_s = p[0] / 0.01;
	double y_s = p[1] / 0.01;
	if (component == 0)
	{
	  if (1. - std::sqrt(x_s*x_s+y_s*y_s) < 0.)
		return 0.;
	  else
		return std::exp(-x_s*x_s-y_s*y_s) * (1. - x_s*x_s - y_s*y_s);
	}
	else if (component == 1)
		return 0.;
}

template <int dim>
void InitialValues2D<dim>::vector_value (const Point<dim> &p,
				    Vector<double>   &values) const 
{
  for (unsigned int c=0; c<2; ++c)
    values(c) = InitialValues2D<dim>::value(p, c);
}

template<int dim>
class Solution: public Function<dim> {
public:
	Solution() : Function<dim>(2*dim) {}

	virtual double value(const Point<dim> &p,
			const unsigned int component = 0) const override;

	virtual void vector_value (const Point<dim> &p, 
			     Vector<double>   &value) const;
};

template<int dim>
double Solution<dim>::value(const Point<dim> &p,
		const unsigned int component) const {
	const double t = this->get_time();
	switch (dim) {
	case 1:
	{
		if (component == 0)
		  return std::sin(t) * p[0] * (1. - p[0]); // u(t,x) = sin(t)x(1-x)
		else if (component == 1)
		  return std::cos(t) * p[0] * (1. - p[0]); // v(t,x) = cos(t)x(1-x)
	}
	case 2:
	{
		return 0.;
	}
	case 3:
	{
		return 0.;
	}
	default:
	{
		Assert(false, ExcNotImplemented());
	}
	}
	return -1.0; // to avoid "no return warning"
}

template <int dim>
void Solution<dim>::vector_value (const Point<dim> &p,
				    Vector<double>   &values) const 
{
  for (unsigned int c=0; c<2*dim; ++c)
    values(c) = Solution<dim>::value(p, c);
}

template<int dim>
class RightHandSide: public Function<dim> {
public:
	RightHandSide() :
			Function<dim>(dim) {
	}
	virtual double value(const Point<dim> &p,
			const unsigned int component = 0) const;
};

template<int dim>
double RightHandSide<dim>::value(const Point<dim> &p,
		const unsigned int component) const {
	const double t = this->get_time();
	//return 1.;

	switch (dim) {
	case 1:
		return (2. - p[0]*(1.-p[0])) * std::sin(t);
	case 2:
		return 0.;
	case 3:
	{
		double max_force = 0.5;
		double time_of_max_force = 5.;
		double time_2_release_force = 1.;
		// bool force_on = (t <= 10.);
		if (component == 0)
			return 0.;
		else if (component == 1)
			if (t <= time_of_max_force)
				return max_force * (t/time_of_max_force);
			else if (t <= time_of_max_force + time_2_release_force)
				return max_force * (1-(t-time_of_max_force)/time_2_release_force);
			else
				return 0.;
		else if (component == 2)
			return 0.;
	}
	default:
	{
		Assert(false, ExcNotImplemented());
	}
	}
}

class Slab {
public:
  // constructor
  Slab(unsigned int r, double start_time, double end_time);
  
  // variables
  Triangulation<1> time_triangulation;
  FE_Q<1>          time_fe;
  DoFHandler<1>    time_dof_handler;
  
  double start_time, end_time;
};

Slab::Slab(unsigned int r, double start_time, double end_time) :
    time_fe(r), time_dof_handler(time_triangulation), start_time(start_time), end_time(end_time) {
}  

template<int dim>
class SpaceTime {
public:
	SpaceTime(std::string problem_name, int s, std::vector<unsigned int> r, std::vector<double> time_points = {0., 1.},
		unsigned int max_n_refinement_cycles = 3, bool refine_space = true, bool refine_time = true, bool split_slabs = true);
	void run();

private:
	void make_grids();
	void setup_system(std::shared_ptr<Slab> &slab, unsigned int k);
	Tensor<1,dim> compute_functional_values(std::shared_ptr<Slab> &slab, Vector<double> &slab_u);
	Tensor<1,dim> compute_normal_stress(Vector<double> space_solution);

	std::string problem_name;
	
	// space
	Triangulation<dim>            space_triangulation;		
	FESystem<dim>                 space_fe;
	DoFHandler<dim>               space_dof_handler;

	// time
	std::vector< std::shared_ptr<Slab> > slabs;
		
	// space-time
	SparsityPattern sparsity_pattern;
	SparseMatrix<double> system_matrix;
	SparseMatrix<double> mass_matrix;
	Vector<double> solution;
	Vector<double> initial_solution; // u(0) or u(t_0)
	Vector<double> system_rhs;
	Vector<double> dual_rhs;

	FEValuesExtractors::Vector displacement;
  	FEValuesExtractors::Vector velocity;
	types::global_dof_index n_space_u;
	types::global_dof_index n_space_v;

	double start_time, end_time;
	std::set< std::pair<double, unsigned int> > time_support_points; // (time_support_point, support_point_index)
	unsigned int n_snapshots;
	unsigned int max_n_refinement_cycles;
	bool refine_space, refine_time, split_slabs;
	double L2_error;
	std::vector<double> L2_error_vals;
	unsigned int n_active_time_cells;
	unsigned int n_time_dofs;
	ConvergenceTable convergence_table;
	
	double E  = 1000.0;
	double nu = 0.3;
	double mu = E / (2.*(1. + nu));
	double lambda = E*nu / ((1. + nu)*(1. - 2.*nu));
};

template<int dim>
SpaceTime<dim>::SpaceTime(std::string problem_name, int s, std::vector<unsigned int> r, std::vector<double> time_points,
		unsigned int max_n_refinement_cycles, bool refine_space, bool refine_time, 
                bool split_slabs) :
		 problem_name(problem_name),
		 space_fe(/*u*/ FE_Q<dim> (s),dim, /*v*/ FE_Q<dim> (s),dim),
		 space_dof_handler(space_triangulation),
		 max_n_refinement_cycles(max_n_refinement_cycles),
		 refine_space(refine_space), refine_time(refine_time),
		 split_slabs(split_slabs) {
    // time_points = [t_0, t_1, ..., t_M]
    // r = [r_1, r_2, ..., r_M] with r_k is the temporal FE degree on I_k = (t_{k-1},t_k]
    Assert(r.size()+1 == time_points.size(), ExcDimensionMismatch(r.size()+1, time_points.size()));
    
    // create slabs
    for (unsigned int k = 0; k < r.size(); ++k)
      slabs.push_back(std::make_shared<Slab> (r[k], time_points[k], time_points[k+1]));
    
    start_time = time_points[0];
    end_time   = time_points[time_points.size()-1];
}

template<>
void SpaceTime<3>::make_grids() {
	// Example from https://fenicsproject.org/olddocs/dolfin/2019.1.0/python/demos/elastodynamics/demo_elastodynamics.py.html

	// create grids
	GridGenerator::subdivided_hyper_rectangle(space_triangulation, {6,1,1}, Point<3>(0.,0.,0.), Point<3>(6., 1., 1.));
	for (auto &slab : slabs)
	  GridGenerator::hyper_rectangle(slab->time_triangulation, Point<1>(slab->start_time), Point<1>(slab->end_time));

	for (auto &cell : space_triangulation.cell_iterators())
	    for (unsigned int face = 0; face < GeometryInfo<3>::faces_per_cell;face++)
		    if (cell->face(face)->at_boundary())
		    {
		    	if (cell->face(face)->center()[0] < 1e-10)
					cell->face(face)->set_boundary_id(0); //   hom. Dirichlet BC
		    	else if (cell->face(face)->center()[1] > (1. - 1.e-10))
					cell->face(face)->set_boundary_id(2); // inhom. Neumann BC
				else
					cell->face(face)->set_boundary_id(1); //   hom. Neumann BC
		    }

	// globally refine the grids
	space_triangulation.refine_global(0);
	for (auto &slab : slabs)
	  slab->time_triangulation.refine_global(0);
}

template<int dim>
void SpaceTime<dim>::setup_system(std::shared_ptr<Slab> &slab, unsigned int k) {
	std::cout << "Slab Q_" << k << " = Ω x (" << slab->start_time  << "," << slab->end_time << "):" << std::endl;
	std::cout << "   Finite Elements: cG(" << space_fe.degree << ") (space), cG("
		  << slab->time_fe.get_degree() << ") (time)" << std::endl;

	slab->time_dof_handler.distribute_dofs(slab->time_fe);

	std::cout << "   Number of active cells: " << space_triangulation.n_active_cells() << " (space), "
		  << slab->time_triangulation.n_active_cells() << " (time)" << std::endl;
	std::cout << "   Number of degrees of freedom: " << space_dof_handler.n_dofs() 
	          << " (" << n_space_u << '+' << n_space_v <<  ')' << " (space), "
		  << slab->time_dof_handler.n_dofs() << " (time)" << std::endl;
  
	/////////////////////////////////////////////////////////////////////////////////////////
	// space-time sparsity pattern = tensor product of spatial and temporal sparsity pattern
	//
	
		// renumber temporal DoFs, which for cG(2) are numbered 0, 2, 1 and not 0, 1, 2
	// time
	std::vector<Point<1>> _time_support_points(slab->time_dof_handler.n_dofs());
	DoFTools::map_dofs_to_support_points(
		MappingQ1<1,1>(),                    
		slab->time_dof_handler,
		_time_support_points
	); // {0., 10., 5.}

	std::map<double, types::global_dof_index, std::less<double>> time_support_map;
	for (unsigned int i = 0; i < _time_support_points.size(); ++i)
		time_support_map[_time_support_points[i][0]] = i;

	// sort time_support_points by time
	std::sort(std::begin(_time_support_points), std::end(_time_support_points), [](Point<1> const& a, Point<1> const& b) { return a[0] < b[0]; });

	std::vector<types::global_dof_index> time_support_points_order_new(_time_support_points.size());
	for (unsigned int i = 0; i < _time_support_points.size(); ++i)
		time_support_points_order_new[time_support_map[_time_support_points[i][0]]] = i;

	slab->time_dof_handler.renumber_dofs(time_support_points_order_new);

	std::vector<Point<1>> time_support_points_new(slab->time_dof_handler.n_dofs());
	DoFTools::map_dofs_to_support_points(
		MappingQ1<1,1>(),                    
		slab->time_dof_handler,
		time_support_points_new
	); // {0., 5., 10.}

	// for debugging:
	// for (auto& t : time_support_points_new)
	// 	std::cout << "t = " << t[0] << std::endl;


	if (k == 1)
	{
	  // spatial sparsity pattern
	  DynamicSparsityPattern space_dsp(space_dof_handler.n_dofs());
	  DoFTools::make_sparsity_pattern(space_dof_handler, space_dsp);

	  SparsityPattern space_sparsity_pattern;
	  space_sparsity_pattern.copy_from(space_dsp);
	  std::ofstream out_space_sparsity("space_sparsity_pattern.svg");
	  space_sparsity_pattern.print_svg(out_space_sparsity);

	  // temporal sparsity pattern
	  DynamicSparsityPattern time_dsp(slab->time_dof_handler.n_dofs());
	  DoFTools::make_sparsity_pattern(slab->time_dof_handler, time_dsp);
	  
	  // space-time sparsity pattern
	  DynamicSparsityPattern dsp(space_dof_handler.n_dofs() * slab->time_dof_handler.n_dofs());
	  for (auto &space_entry : space_dsp)
	    for (auto &time_entry : time_dsp)
	      dsp.add(
		space_entry.row()    + space_dof_handler.n_dofs() * time_entry.row(),   // test  function
		space_entry.column() + space_dof_handler.n_dofs() * time_entry.column() // trial function
	      );
	  
	  sparsity_pattern.copy_from(dsp);
	  std::ofstream out_sparsity("sparsity_pattern.svg");
	  sparsity_pattern.print_svg(out_sparsity);
	  
	  system_matrix.reinit(sparsity_pattern);
	  mass_matrix.reinit(sparsity_pattern);
	}
	
	solution.reinit(space_dof_handler.n_dofs() * slab->time_dof_handler.n_dofs());
	system_rhs.reinit(space_dof_handler.n_dofs() * slab->time_dof_handler.n_dofs());
	dual_rhs.reinit(space_dof_handler.n_dofs() * slab->time_dof_handler.n_dofs());
}

template<int dim>
Tensor<1, dim> SpaceTime<dim>::compute_functional_values(std::shared_ptr<Slab> &slab, Vector<double> &slab_u)
{
	Tensor<1, dim> normal_stress;

	FEValues<1> time_fe_values(slab->time_fe, QGauss<1>(slab->time_fe.degree + 6), update_values | update_quadrature_points | update_JxW_values);
	std::vector<types::global_dof_index> time_local_dof_indices(slab->time_fe.n_dofs_per_cell());	

	for (const auto &time_cell : slab->time_dof_handler.active_cell_iterators()) {
	  time_fe_values.reinit(time_cell);
	  time_cell->get_dof_indices(time_local_dof_indices);

	  for (const unsigned int qq : time_fe_values.quadrature_point_indices()) 
	  {
	    // time quadrature point
	    double t_qq = time_fe_values.quadrature_point(qq)[0];
	    
	    // get the FEM space solution at the quadrature point
	    Vector<double> space_solution(space_dof_handler.n_dofs());
		for (const unsigned int ii : time_fe_values.dof_indices())
		{
		  for (unsigned int i = 0; i < space_dof_handler.n_dofs(); ++i)
			space_solution(i) += slab_u(i + time_local_dof_indices[ii] * space_dof_handler.n_dofs()) * time_fe_values.shape_value(ii, qq);
		}

		Tensor<1, dim> normal_stress_qq = compute_normal_stress(space_solution);
	    
	    // add local contributions to global QoI
		normal_stress += normal_stress_qq * time_fe_values.JxW(qq);
	  }
	}
	return normal_stress;
}

template<int dim>
Tensor<1,dim> SpaceTime<dim>::compute_normal_stress(Vector<double> space_solution)
{
	Tensor<1, dim> normal_stress;

	QGauss<dim-1> space_face_quad_formula(space_fe.degree + 4);
	FEFaceValues<dim> space_fe_face_values(space_fe, space_face_quad_formula,
						update_values | update_gradients | update_normal_vectors | update_JxW_values | update_quadrature_points);

	const unsigned int space_dofs_per_cell = space_fe.n_dofs_per_cell();
	const unsigned int n_face_q_points = space_face_quad_formula.size();

	std::vector<types::global_dof_index> space_local_dof_indices(space_dofs_per_cell);
	std::vector<std::vector<Tensor<1, dim>>> face_solution_grads(n_face_q_points, std::vector<Tensor<1, dim>>(2*dim));

	Tensor <2,dim> identity;
	for (unsigned int k=0; k<dim; ++k)
		identity[k][k] = 1.;

	for (const auto &space_cell : space_dof_handler.active_cell_iterators()) {
		for (const unsigned int space_face : space_cell->face_indices())
			if (space_cell->at_boundary(space_face) && (space_cell->face(space_face)->boundary_id() == 0)) // face is at hom. Dirichlet boundary
			{
				space_fe_face_values.reinit(space_cell, space_face);
				space_fe_face_values.get_function_gradients(space_solution, face_solution_grads);

				for (const unsigned int q : space_fe_face_values.quadrature_point_indices())
				{
					Tensor<2, dim> symm_grad_u;
					for (unsigned int l = 0; l < dim; l++)
						for (unsigned int m = 0; m < dim; m++)
							symm_grad_u[l][m] = 0.5 * (face_solution_grads[q][l][m] + face_solution_grads[q][m][l]);

					const Tensor<2, dim> stress_tensor = 2. * mu * symm_grad_u + lambda * trace(symm_grad_u) * identity;

					normal_stress += stress_tensor * space_fe_face_values.normal_vector(q) * space_fe_face_values.JxW(q);
				}
			}
	}

	return normal_stress;
}

template<int dim>
void SpaceTime<dim>::run() {
	std::cout << "Starting problem " << problem_name << " in " << dim << "+1D..." << std::endl;

	displacement = FEValuesExtractors::Vector(0);
	velocity = FEValuesExtractors::Vector(dim);
	
	// create a coarse grid
	make_grids();

	// Refinement loop
	for (unsigned int cycle = 0; cycle < max_n_refinement_cycles; ++cycle) {
		std::cout
				<< "-------------------------------------------------------------"
				<< std::endl;
		std::cout << "|                REFINEMENT CYCLE: " << cycle;
		std::cout << "                         |" << std::endl;
		std::cout
				<< "-------------------------------------------------------------"
				<< std::endl;

		// create output directory if necessary
		std::string dim_dir = "../Data/" + std::to_string(dim) + "D/";
		std::string problem_dir = dim_dir + problem_name + "/";
		std::string output_dir = problem_dir + "cycle=" + std::to_string(cycle) + "/";
		for (auto dir : {"../Data/", dim_dir.c_str(), problem_dir.c_str(), output_dir.c_str()})
		  mkdir(dir, S_IRWXU);

		std::string readin_dir = "../Data/ROM/cycle=" + std::to_string(cycle) + "/error_estimator/";

		// remove old logfiles
		std::remove("python_goal_func_error.txt");
		std::remove("cpp_goal_func_error.txt");

		////////////////////////////////////////////
		// initial value: u(0)
		//
		space_dof_handler.distribute_dofs(space_fe);

		// Renumber spatials DoFs into displacement and velocity DoFs
		// We are dealing with 2*dim components:
		// displacements [0 -- dim-1]:         0
		// velocities  [dim -- 2*dim-1]:       1
		std::vector<unsigned int> block_component (2*dim,0);
		for (unsigned int i=0; i<dim; ++i)
			block_component[dim+i] = 1;
	  	DoFRenumbering::component_wise (space_dof_handler, block_component);

	  	// two blocks: displacement and velocity
		const std::vector<types::global_dof_index> dofs_per_block = DoFTools::count_dofs_per_fe_block(space_dof_handler, block_component);
		n_space_u = dofs_per_block[0];
		n_space_v = dofs_per_block[1];

		// no hanging node constraints 
		AffineConstraints<double> constraints;
		constraints.close();

		// compute initial value vector
		initial_solution.reinit(space_dof_handler.n_dofs());

		///////////////////////////////////////////////////////////////////////////
		// loading in vectors for full-order and reduced goal functional values
		std::vector<double> J_h(slabs.size());
		std::vector<double> J_r(slabs.size());

		// loading in J_h
		{
			std::ifstream J_h_in((readin_dir + "fullorder_goal_func_vals.txt").c_str(), std::ios_base::in);
			std::string _line;
			unsigned int i = 0;
			while (std::getline (J_h_in, _line))
				J_h[i++] = std::stod(_line);
		}

		// loading in J_r
		{
			std::ifstream J_r_in((readin_dir + "reduced_goal_func_vals.txt").c_str(), std::ios_base::in);
			std::string _line;
			unsigned int i = 0;
			while (std::getline (J_r_in, _line))
				J_r[i++] = std::stod(_line);
		}

		//////////////////////////////////////////////////////////////
		// loading in slabwise dealii::Vectors for U_r, U_h and Z_h
		unsigned int _r = slabs[0]->time_fe.get_degree()+1;
		std::vector< Vector<double> > U_r(slabs.size(), Vector<double>(_r * space_dof_handler.n_dofs()));
		std::vector< Vector<double> > U_h(slabs.size(), Vector<double>(_r * space_dof_handler.n_dofs()));
		std::vector< Vector<double> > Z_h(slabs.size(), Vector<double>(_r * space_dof_handler.n_dofs()));

        for (unsigned int k = 0; k < slabs.size(); ++k)
		{
			// loading U_r[k]
			{
				std::ifstream U_r_in((readin_dir + "reduced_primal_solution_" +  Utilities::int_to_string(k, 5)  + ".txt").c_str(), std::ios_base::in);
				std::string _line;
				unsigned int i = 0;
				while (std::getline (U_r_in, _line))
					U_r[k][i++] = std::stod(_line);
			}

			// loading U_h[k]
			{
				std::ifstream U_h_in((readin_dir + "fullorder_primal_solution_" +  Utilities::int_to_string(k, 5)  + ".txt").c_str(), std::ios_base::in);
				std::string _line;
				unsigned int i = 0;
				while (std::getline (U_h_in, _line))
					U_h[k][i++] = std::stod(_line);
			}

			// loading Z_h[k]
			{
				std::ifstream Z_h_in((readin_dir + "fullorder_dual_solution_" +  Utilities::int_to_string(k, 5)  + ".txt").c_str(), std::ios_base::in);
				std::string _line;
				unsigned int i = 0;
				while (std::getline (Z_h_in, _line))
					Z_h[k][i++] = std::stod(_line);
			}
		} // end for slab

		// TODO: load U_r, U_h and Z_h as slab vectors for this refinement cycle

		for (unsigned int k = 0; k < slabs.size(); ++k)
		{
			// create and solve linear system
			setup_system(slabs[k], k+1);

            // compute J(U_h) - J(U_r) on current slab
			// TODO: compute J(U_h) and J(U_r) on current slab
            // TODO: compute_functional_values(cycle, k); for U_h
			// TODO: compute_functional_values(cycle, k); for U_r
			double python_goal_func_error = J_h[k] - J_r[k];
			std::cout << "PY: " << python_goal_func_error << std::endl;
			double cpp_goal_func_error = compute_functional_values(slabs[k], U_h[k])[1] - compute_functional_values(slabs[k], U_r[k])[1];
			std::cout << "C++: " << cpp_goal_func_error << std::endl;
			
			// output the values into log files
			std::ofstream python_goal_out("python_goal_func_error.txt", std::ios_base::app);
			std::ofstream cpp_goal_out("cpp_goal_func_error.txt", std::ios_base::app);
			double slab_t = 0.5 * (slabs[k]->start_time + slabs[k]->end_time);
			python_goal_out << slab_t << "," << python_goal_func_error << std::endl;
			cpp_goal_out << slab_t << "," << cpp_goal_func_error << std::endl;

			///////////////////////
			// prepare next slab
			//
			    
			// get initial value for next slab
			unsigned int ii = slabs[k]->time_dof_handler.n_dofs()-1;
			for (unsigned int i = 0; i < space_dof_handler.n_dofs(); ++i)
			  initial_solution(i) = solution(i + ii * space_dof_handler.n_dofs());
		}

		exit(2);

		// refine mesh
		if (cycle < max_n_refinement_cycles - 1)
		{
		  space_triangulation.refine_global(refine_space);
		  if (split_slabs)
		  {
		    std::vector< std::shared_ptr<Slab> > split_slabs;
		    for (auto &slab : slabs)
		    {
		      split_slabs.push_back(std::make_shared<Slab> (slab->time_fe.get_degree(), slab->start_time, 0.5*(slab->start_time+slab->end_time)));
		      split_slabs.push_back(std::make_shared<Slab> (slab->time_fe.get_degree(), 0.5*(slab->start_time+slab->end_time), slab->end_time));
		    }
		    slabs = split_slabs;

		    for (auto &slab : slabs)
		      GridGenerator::hyper_rectangle(slab->time_triangulation, Point<1>(slab->start_time), Point<1>(slab->end_time));
		  }
		  else
		  {
		    for (auto &slab : slabs)
		      slab->time_triangulation.refine_global(refine_time);
		  }
		}
	}
}

int main() {
	try {
		deallog.depth_console(2);

		const std::string problem_name = "Rod";
		const int dim = 3;

		std::vector<unsigned int> r;
		std::vector<double> t = {0.};
		double T = 40.;
		int N = 1600; //0; //2*4;
		double dt = T / N;
		for (unsigned int i = 0; i < N; ++i)
		{
				r.push_back(2);
				t.push_back((i+1)*dt);
		}

		// run the simulation
		SpaceTime<dim> space_time_problem(
		  problem_name,     // problem_name
		  1,                // s ->  spatial FE degree
		  r,                // r -> temporal FE degree
		  t, 		    	// time_points,
		  4,                // max_n_refinement_cycles,
		  true,             // refine_space
		  false,             // refine_time
		  false              // split_slabs
		);
		space_time_problem.run();

	} catch (std::exception &exc) {
		std::cerr << std::endl << std::endl
				<< "----------------------------------------------------"
				<< std::endl;
		std::cerr << "Exception on processing: " << std::endl << exc.what()
				<< std::endl << "Aborting!" << std::endl
				<< "----------------------------------------------------"
				<< std::endl;

		return 1;
	} catch (...) {
		std::cerr << std::endl << std::endl
				<< "----------------------------------------------------"
				<< std::endl;
		std::cerr << "Unknown exception!" << std::endl << "Aborting!"
				<< std::endl
				<< "----------------------------------------------------"
				<< std::endl;
		return 1;
	}

	return 0;
}
