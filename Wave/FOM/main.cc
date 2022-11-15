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
 * TENSOR-PRODUCT SPACE-TIME FINITE ELEMENTS:
 * ==========================================
 * Tensor-product space-time code for the wave equation with Q^s finite elements in space and Q^r finite elements in time: cG(s)cG(r)
 *
 * Author: Julian Roth, 2022
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
#include <sys/stat.h> // for mkdir

using namespace dealii;

template<int dim>
class InitialValues: public Function<dim> {
public:
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
class Solution: public Function<dim> {
public:
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
		Assert(false, ExcNotImplemented());
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
  for (unsigned int c=0; c<2; ++c)
    values(c) = Solution<dim>::value(p, c);
}

template<int dim>
class RightHandSide: public Function<dim> {
public:
	RightHandSide() :
			Function<dim>() {
	}
	virtual double value(const Point<dim> &p,
			const unsigned int component = 0) const;
};

template<int dim>
double RightHandSide<dim>::value(const Point<dim> &p,
		const unsigned int /*component*/) const {
	const double t = this->get_time();
	//return 1.;

	switch (dim) {
	case 1:
		return (2. - p[0]*(1.-p[0])) * std::sin(t);
	default:
	{
		Assert(false, ExcNotImplemented());
	}
	}

}

template<int dim>
class SpaceTime {
public:
	SpaceTime(int s, int r, double start_time = 0.0, double end_time = 1.0,
		unsigned int max_n_refinement_cycles = 3, bool refine_space = true, bool refine_time = true);
	void run();
	void print_grids(std::string file_name_space, std::string file_name_time);
	void print_convergence_table();

private:
	void make_grids();
	void setup_system();
	void assemble_system();
	void apply_initial_condition();
	void apply_boundary_conditions();
	void solve();
	void output_results(const unsigned int refinement_cycle) const;
	void process_solution(const unsigned int cycle);
	
	// space
	Triangulation<dim>            space_triangulation;		
	FESystem<dim>                space_fe;
	DoFHandler<dim>               space_dof_handler;

	// time
	Triangulation<1> time_triangulation;
	FE_Q<1>          time_fe;
	DoFHandler<1>    time_dof_handler;
		
	// space-time
	SparsityPattern sparsity_pattern;
	SparseMatrix<double> system_matrix;
	Vector<double> solution;
	Vector<double> system_rhs;

	const FEValuesExtractors::Scalar displacement = 0;
  	const FEValuesExtractors::Scalar velocity     = 1;
	types::global_dof_index n_space_u;
	types::global_dof_index n_space_v;

	double start_time, end_time;
	std::set< std::pair<double, unsigned int> > time_support_points; // (time_support_point, support_point_index)
	unsigned int max_n_refinement_cycles;
	bool refine_space, refine_time;
	std::vector<double> L2_error_vals;
	ConvergenceTable convergence_table;
};

template<int dim>
SpaceTime<dim>::SpaceTime(int s, int r, double start_time, double end_time,
		unsigned int max_n_refinement_cycles, bool refine_space, bool refine_time) :
		 space_fe(/*u*/ FE_Q<1> (s),1, /*v*/ FE_Q<1> (s),1),
		 space_dof_handler(space_triangulation),
		 time_fe(r), time_dof_handler(time_triangulation),
		 start_time(start_time), end_time(end_time), 
		 max_n_refinement_cycles(max_n_refinement_cycles),
		 refine_space(refine_space), refine_time(refine_time) {
}

void print_1d_grid(std::ofstream &out, Triangulation<1> &triangulation, double start, double end) {
	out << "<svg width='1200' height='200' xmlns='http://www.w3.org/2000/svg' version='1.1'>" << std::endl;
	out << "<rect fill='white' width='1200' height='200'/>" << std::endl;
	out << "  <line x1='100' y1='100' x2='1100' y2='100' style='stroke:black;stroke-width:4'/>" << std::endl; // timeline
	out << "  <line x1='100' y1='125' x2='100' y2='75' style='stroke:black;stroke-width:4'/>" << std::endl; // first tick

	// ticks
	for (auto &cell : triangulation.active_cell_iterators())
		out << "  <line x1='" << (int)(1000*(cell->vertex(1)[0]-start)/(end-start)) + 100 <<"' y1='125' x2='" 
		    << (int)(1000*(cell->vertex(1)[0]-start)/(end-start)) + 100 <<"' y2='75' style='stroke:black;stroke-width:4'/>" << std::endl;

	out << "</svg>" << std::endl;
}

template<>
void SpaceTime<2>::print_grids(std::string file_name_space, std::string file_name_time) {
	// space	
	std::ofstream out_space(file_name_space);
	GridOut grid_out_space;
	grid_out_space.write_svg(space_triangulation, out_space);
		
	// time	
	std::ofstream out_time(file_name_time);
	print_1d_grid(out_time, time_triangulation, start_time, end_time);
}

template<>
void SpaceTime<1>::print_grids(std::string file_name_space, std::string file_name_time) {
	// space	
	std::ofstream out_space(file_name_space);
	print_1d_grid(out_space, space_triangulation, 0., 1.);
		
	// time	
	std::ofstream out_time(file_name_time);
	print_1d_grid(out_time, time_triangulation, start_time, end_time);
}

template<>
void SpaceTime<1>::make_grids() {
	// create grids
	GridGenerator::hyper_rectangle(space_triangulation, Point<1>(0.), Point<1>(1.));
	GridGenerator::hyper_rectangle(time_triangulation, Point<1>(start_time), Point<1>(end_time));
	
	// ensure that both points of the boundary have boundary_id == 0
	for (auto &cell : space_triangulation.cell_iterators())
	    for (unsigned int face = 0; face < GeometryInfo<1>::faces_per_cell;face++)
		    if (cell->face(face)->at_boundary())
			  cell->face(face)->set_boundary_id(0);

	// globally refine the grids
	space_triangulation.refine_global(2);
	time_triangulation.refine_global(0); //2);
}

template<>
void SpaceTime<2>::make_grids() {
	// Hartmann test problem

	// create grids
	GridGenerator::hyper_rectangle(space_triangulation, Point<2>(0., 0.), Point<2>(1., 1.));
	GridGenerator::hyper_rectangle(time_triangulation, Point<1>(start_time), Point<1>(end_time));

	// globally refine the grids
	space_triangulation.refine_global(1);
	time_triangulation.refine_global(1);
}

template<int dim>
void SpaceTime<dim>::setup_system() {
	space_dof_handler.distribute_dofs(space_fe);
	time_dof_handler.distribute_dofs(time_fe);

	// Renumber spatials DoFs into displacement and velocity DoFs
  	DoFRenumbering::component_wise (space_dof_handler, {0, 1});

  	// two blocks: displacement and velocity
	const std::vector<types::global_dof_index> dofs_per_block = DoFTools::count_dofs_per_fe_block(space_dof_handler, {0, 1});
	n_space_u = dofs_per_block[0];
	n_space_v = dofs_per_block[1];

	std::cout << "Number of active cells: " << space_triangulation.n_active_cells() << " (space), "
		  << time_triangulation.n_active_cells() << " (time)" << std::endl;
	std::cout << "Number of degrees of freedom: " << space_dof_handler.n_dofs() 
	          << " (" << n_space_u << '+' << n_space_v <<  ')' << " (space), "
		  << time_dof_handler.n_dofs() << " (time)" << std::endl;
  
	/////////////////////////////////////////////////////////////////////////////////////////
	// space-time sparsity pattern = tensor product of spatial and temporal sparsity pattern
	//
		 
	// spatial sparsity pattern
	DynamicSparsityPattern space_dsp(space_dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(space_dof_handler, space_dsp);

	SparsityPattern space_sparsity_pattern;
	space_sparsity_pattern.copy_from(space_dsp);
	std::ofstream out_space_sparsity("space_sparsity_pattern.svg");
	space_sparsity_pattern.print_svg(out_space_sparsity);

	// temporal sparsity pattern
	DynamicSparsityPattern time_dsp(time_dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(time_dof_handler, time_dsp);
	
	// space-time sparsity pattern
	DynamicSparsityPattern dsp(space_dof_handler.n_dofs() * time_dof_handler.n_dofs());
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
	
	solution.reinit(space_dof_handler.n_dofs() * time_dof_handler.n_dofs());
	system_rhs.reinit(space_dof_handler.n_dofs() * time_dof_handler.n_dofs());
}

template<int dim>
void SpaceTime<dim>::assemble_system() {
	RightHandSide<dim> right_hand_side;

	// space
	QGauss<dim> space_quad_formula(space_fe.degree + 2);
	FEValues<dim> space_fe_values(space_fe, space_quad_formula,
			update_values | update_gradients | update_quadrature_points
					| update_JxW_values);
	const unsigned int space_dofs_per_cell = space_fe.n_dofs_per_cell();
	std::vector<types::global_dof_index> space_local_dof_indices(space_dofs_per_cell);
	
	// time
	QGauss<1> time_quad_formula(time_fe.degree + 2);
	FEValues<1> time_fe_values(time_fe, time_quad_formula,
			update_values | update_gradients | update_quadrature_points
					| update_JxW_values);
	const unsigned int time_dofs_per_cell = time_fe.n_dofs_per_cell();
	std::vector<types::global_dof_index> time_local_dof_indices(time_dofs_per_cell);
	
	// local contributions on space-time cell
	FullMatrix<double> cell_matrix(space_dofs_per_cell * time_dofs_per_cell, space_dofs_per_cell * time_dofs_per_cell);
	Vector<double> cell_rhs(space_dofs_per_cell * time_dofs_per_cell);
	std::vector<types::global_dof_index> local_dof_indices(space_dofs_per_cell * time_dofs_per_cell);

	// locally assemble on each space-time cell
	for (const auto &space_cell : space_dof_handler.active_cell_iterators()) {
	  space_fe_values.reinit(space_cell);
	  space_cell->get_dof_indices(space_local_dof_indices);
	  for (const auto &time_cell : time_dof_handler.active_cell_iterators()) {
	    time_fe_values.reinit(time_cell);
	    time_cell->get_dof_indices(time_local_dof_indices);
	    
	    cell_matrix = 0;
	    cell_rhs = 0;
	    
	    for (const unsigned int qq : time_fe_values.quadrature_point_indices())
	    {
	      // time quadrature point
	      const double t_qq = time_fe_values.quadrature_point(qq)[0];
	      right_hand_side.set_time(t_qq);
	      
	      for (const unsigned int q : space_fe_values.quadrature_point_indices())
	      {
		// space quadrature point
		const auto x_q = space_fe_values.quadrature_point(q);
		
		for (const unsigned int i : space_fe_values.dof_indices())
		  for (const unsigned int ii : time_fe_values.dof_indices())
		  {
		    // right hand side
		    cell_rhs(i + ii * space_dofs_per_cell) += (
		         space_fe_values[displacement].value(i, q) * time_fe_values.shape_value(ii, qq) * // ϕ_{i,ii}(t_qq, x_q)
								     right_hand_side.value(x_q) * // f(t_qq, x_q)
					        space_fe_values.JxW(q) * time_fe_values.JxW(qq)   // d(t,x)
		    );
		    
		    // system matrix
		    for (const unsigned int j : space_fe_values.dof_indices())
		      for (const unsigned int jj : time_fe_values.dof_indices())
			cell_matrix(
			  j + jj * space_dofs_per_cell,
			  i + ii * space_dofs_per_cell
			) += (
			  space_fe_values[velocity].value(i, q) * time_fe_values.shape_grad(ii, qq)[0] *     // ∂_t ϕ^v_{i,ii}(t_qq, x_q)
			  space_fe_values[displacement].value(j, q) * time_fe_values.shape_value(jj, qq)     //     ϕ^u_{j,jj}(t_qq, x_q)
												  // +
			  + space_fe_values[displacement].gradient(i, q) * time_fe_values.shape_value(ii, qq) *  // ∇_x ϕ^u_{i,ii}(t_qq, x_q)
			  space_fe_values[displacement].gradient(j, q) * time_fe_values.shape_value(jj, qq)      // ∇_x ϕ^u_{j,jj}(t_qq, x_q)
												  // +
			  + space_fe_values[displacement].value(i, q) * time_fe_values.shape_grad(ii, qq)[0] * // ∂_t ϕ^u_{i,ii}(t_qq, x_q)
			  space_fe_values[velocity].value(j, q) * time_fe_values.shape_value(jj, qq)           //     ϕ^v_{j,jj}(t_qq, x_q)
												  // -
			  - space_fe_values[velocity].value(i, q) * time_fe_values.shape_value(ii, qq) *  //  ϕ^v_{i,ii}(t_qq, x_q)
			  space_fe_values[velocity].value(j, q) * time_fe_values.shape_value(jj, qq)      //  ϕ^v_{j,jj}(t_qq, x_q)
			) * space_fe_values.JxW(q) * time_fe_values.JxW(qq); 			  // d(t,x)
		  }
	      }
	    }
	    
	    // distribute local to global
	    for (const unsigned int i : space_fe_values.dof_indices())
	      for (const unsigned int ii : time_fe_values.dof_indices())
	      {
		// right hand side
		system_rhs(space_local_dof_indices[i] + time_local_dof_indices[ii] * space_dof_handler.n_dofs()) += cell_rhs(i + ii * space_dofs_per_cell);
		
		// system matrix
		for (const unsigned int j : space_fe_values.dof_indices())
		  for (const unsigned int jj : time_fe_values.dof_indices())
		    system_matrix.add(
		      space_local_dof_indices[i] + time_local_dof_indices[ii] * space_dof_handler.n_dofs(),
		      space_local_dof_indices[j] + time_local_dof_indices[jj] * space_dof_handler.n_dofs(),
		      cell_matrix(i + ii * space_dofs_per_cell, j + jj * space_dofs_per_cell)
		    );
	      }
	  }
	}
	
	apply_initial_condition();
	apply_boundary_conditions();
}

template<int dim>
void SpaceTime<dim>::apply_initial_condition() {
    // apply initial value on linear system
  
    // no hanging node constraints ´
    AffineConstraints<double> constraints;
    constraints.close();
  
    // compute initial value vector
    Vector<double> initial_solution(space_dof_handler.n_dofs());
    VectorTools::interpolate(space_dof_handler,
                         InitialValues<dim>(),
                         initial_solution,
			 ComponentMask());
  
    // clear first N_space rows of system matrix
    for (unsigned int i = 0; i < space_dof_handler.n_dofs(); ++i)
    {
      // iterate over entries of i.th row of system matrix
      // A_ij = δ_ij
      for (typename SparseMatrix<double>::iterator p = system_matrix.begin(i); p != system_matrix.end(i); ++p)
         p->value() = 0.;
      system_matrix.set(i, i, 1.);
      system_rhs(i) = initial_solution(i);
    }

}	

template<int dim>
void SpaceTime<dim>::apply_boundary_conditions() {
   // apply the spatial Dirichlet boundary conditions at each temporal DoF
   Solution<dim> solution_func;
   
   // remove old temporal support points
   time_support_points.clear();
   
   FEValues<1> time_fe_values(time_fe, Quadrature<1>(time_fe.get_unit_support_points()), update_quadrature_points);
   std::vector<types::global_dof_index> time_local_dof_indices(time_fe.n_dofs_per_cell());
   
   for (const auto &time_cell : time_dof_handler.active_cell_iterators()) {
    time_fe_values.reinit(time_cell);
    time_cell->get_dof_indices(time_local_dof_indices);
    
    // using temporal support points as quadrature points
    for (const unsigned int qq : time_fe_values.quadrature_point_indices()) 
    {
	// time quadrature point
	double t_qq = time_fe_values.quadrature_point(qq)[0];
	solution_func.set_time(t_qq);
	time_support_points.insert(std::make_pair(t_qq, time_local_dof_indices[qq]));
	
	// determine spatial boundary values at temporal support point
	std::map<types::global_dof_index, double> boundary_values;
	VectorTools::interpolate_boundary_values(space_dof_handler, 0, solution_func/*ZeroFunction<dim>(1+1)*/, boundary_values);
	
	// calculate the correct space-time entry and apply the Dirichlet BC
	for (auto &entry : boundary_values)
	{
	  types::global_dof_index id = entry.first + time_local_dof_indices[qq] * space_dof_handler.n_dofs();
	  
	  // apply BC
	  for (typename SparseMatrix<double>::iterator p = system_matrix.begin(id); p != system_matrix.end(id); ++p)
	    p->value() = 0.;
	  system_matrix.set(id, id, 1.);
	  system_rhs(id) = entry.second;
	}
      }
    }
}

template<int dim>
void SpaceTime<dim>::solve() {
	SparseDirectUMFPACK A_direct;
	A_direct.initialize(system_matrix);
	A_direct.vmult(solution, system_rhs);
}

template<>
void SpaceTime<1>::output_results(const unsigned int refinement_cycle) const {
	// create output directory if necessary
	std::string output_dir = "output/dim=1/cycle=" + std::to_string(refinement_cycle) + "/";
	for (auto dir : {"output/", "output/dim=1/", output_dir.c_str()})
	  mkdir(dir, S_IRWXU);

	//////////////////////////////////////////////////////////////////////////////
	// output space-time vectors for displacement and velocity separately as .txt
	//
	Vector<double> solution_u(n_space_u * time_dof_handler.n_dofs());
	Vector<double> solution_v(n_space_v * time_dof_handler.n_dofs());
	unsigned int index_u = 0;
	unsigned int index_v = 0;
	for (unsigned int ii = 0; ii < time_dof_handler.n_dofs(); ++ii)
	{
          // displacement
	  for (unsigned int j=ii*space_dof_handler.n_dofs(); j < ii*space_dof_handler.n_dofs()+n_space_u; ++j)
  	  {
	     solution_u[index_u] = solution[j];
	     index_u++;
	  }
	  // velocity
	  for (unsigned int j=ii*space_dof_handler.n_dofs()+n_space_u; j < (ii+1)*space_dof_handler.n_dofs(); ++j)
  	  {
	     solution_v[index_v] = solution[j];
	     index_v++;
	  }
	}
	std::ofstream solution_u_out(output_dir + "solution_u.txt");
	solution_u.print(solution_u_out, /*precision*/16);
	std::ofstream solution_v_out(output_dir + "solution_v.txt");
	solution_v.print(solution_v_out, /*precision*/16);

	///////////////////////////////////////////
	// output coordinates for t and x as .txt
	//

	// space
	std::vector<Point<1>> space_support_points(space_dof_handler.n_dofs());
	DoFTools::map_dofs_to_support_points(
		MappingQ1<1,1>(),                    
		space_dof_handler,
		space_support_points
	); 

	std::ofstream x_out(output_dir + "coordinates_x.txt");
	x_out.precision(9);
	x_out.setf(std::ios::scientific, std::ios::floatfield);

	unsigned int i = 0;
	for (auto point : space_support_points)
	{
	  if (i == n_space_u)
	     break;
	  x_out << point[0] << ' ';
	  i++;
	}
	x_out << std::endl;

	// time
	std::vector<Point<1>> time_support_points(time_dof_handler.n_dofs());
	DoFTools::map_dofs_to_support_points(
		MappingQ1<1,1>(),                    
		time_dof_handler,
		time_support_points
	); 

	std::ofstream t_out(output_dir + "coordinates_t.txt");
	t_out.precision(9);
	t_out.setf(std::ios::scientific, std::ios::floatfield);

	for (auto point : time_support_points)
		t_out << point[0] << ' ';
	t_out << std::endl;
	
	std::string plotting_cmd = "python3 plot_solution.py  " + output_dir;
	system(plotting_cmd.c_str());
}

template<>
void SpaceTime<2>::output_results(const unsigned int refinement_cycle) const {
// TODO: implement output_results for 2+1D wave equation
/*
	// create output directory if necessary
	std::string output_dir = "output/dim=2/cycle=" + std::to_string(refinement_cycle) + "/";
	for (auto dir : {"output/", "output/dim=2/", output_dir.c_str()})
	  mkdir(dir, S_IRWXU);
  
	// output results as VTK files
	unsigned int ii_ordered = 0;
	for (auto time_point : time_support_points)
	{
	  double t_qq = time_point.first;
	  unsigned int ii = time_point.second;
	  
	  DataOut<2> data_out;
	  data_out.attach_dof_handler(space_dof_handler);
	  
	  Vector<double> space_solution(space_dof_handler.n_dofs());
	  for (unsigned int i = 0; i < space_dof_handler.n_dofs(); ++i)
	    space_solution(i) = solution(i + ii * space_dof_handler.n_dofs());
	  
	  // TODO: debug this in 2D
	  std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(1+1, DataComponentInterpretation::component_is_scalar);
	  data_out.add_data_vector(space_solution, {"displacement", "velocity"}, DataOut<2>::type_dof_data, data_component_interpretation);
	  data_out.build_patches();
	 
	  data_out.set_flags(DataOutBase::VtkFlags(t_qq, ii));
	  
	  std::ofstream output(output_dir + "solution" + Utilities::int_to_string(ii_ordered, 5) + ".vtk");
	  data_out.write_vtk(output);
	  
	  ++ii_ordered;
	}
*/
}

template<int dim>
void SpaceTime<dim>::process_solution(const unsigned int cycle) {
	Solution<dim> solution_func;
	double L2_error = 0.;
	
	FEValues<1> time_fe_values(time_fe, QGauss<1>(time_fe.degree + 2), update_values | update_quadrature_points | update_JxW_values);
	std::vector<types::global_dof_index> time_local_dof_indices(time_fe.n_dofs_per_cell());
	
	for (const auto &time_cell : time_dof_handler.active_cell_iterators()) {
	  time_fe_values.reinit(time_cell);
	  time_cell->get_dof_indices(time_local_dof_indices);

	  for (const unsigned int qq : time_fe_values.quadrature_point_indices()) 
	  {
	    // time quadrature point
	    double t_qq = time_fe_values.quadrature_point(qq)[0];
	    solution_func.set_time(t_qq);
	    
	    // get the FEM space solution at the quadrature point
	    Vector<double> space_solution(space_dof_handler.n_dofs());
	    for (const unsigned int ii : time_fe_values.dof_indices())
	    {
	      for (unsigned int i = 0; i < space_dof_handler.n_dofs(); ++i)
		space_solution(i) += solution(i + time_local_dof_indices[ii] * space_dof_handler.n_dofs()) * time_fe_values.shape_value(ii, qq);
	    }

	    // get the analytical space solution at the quadrature point
	    Vector<double> analytical_solution(space_dof_handler.n_dofs());
    	    VectorTools::interpolate(space_dof_handler,
                         solution_func,
                         analytical_solution,
			 ComponentMask());

	    // compute the L2 error at the temporal quadrature point
	    Vector<float> difference_per_cell(space_triangulation.n_active_cells());
	    VectorTools::integrate_difference(space_dof_handler, space_solution, solution_func,
			    difference_per_cell, QGauss<dim>(space_fe.degree + 2),
			    VectorTools::L2_norm);
	    const double L2_error_t_qq = VectorTools::compute_global_error(space_triangulation,
			    difference_per_cell, VectorTools::L2_norm);
	    
	    // add local contributions to global L2 error
	    L2_error += std::pow(L2_error_t_qq, 2) * time_fe_values.JxW(qq);
	  }
	}
	
	L2_error = std::sqrt(L2_error);
	L2_error_vals.push_back(L2_error);
	
	// add values to 
	const unsigned int n_active_cells = space_triangulation.n_active_cells() * time_triangulation.n_active_cells();
	const unsigned int n_space_dofs   = space_dof_handler.n_dofs();
	const unsigned int n_time_dofs    = time_dof_handler.n_dofs();
	const unsigned int n_dofs         = n_space_dofs * n_time_dofs;

	convergence_table.add_value("cycle", cycle);
	convergence_table.add_value("cells", n_active_cells);
	convergence_table.add_value("dofs", n_dofs);
	convergence_table.add_value("dofs(space)", n_space_dofs);
	convergence_table.add_value("dofs(time)", n_time_dofs);
	convergence_table.add_value("L2", L2_error);
}

template<int dim>
void SpaceTime<dim>::print_convergence_table() {
	convergence_table.set_precision("L2", 3);
	convergence_table.set_scientific("L2", true);
	convergence_table.set_tex_caption("cells", "\\# cells");
	convergence_table.set_tex_caption("dofs", "\\# dofs");
	convergence_table.set_tex_caption("dofs(space)", "\\# dofs space");
	convergence_table.set_tex_caption("dofs(time)", "\\# dofs time");
	convergence_table.set_tex_caption("L2", "@f$L^2@f$-error");
	convergence_table.set_tex_format("cells", "r");
	convergence_table.set_tex_format("dofs", "r");
	convergence_table.set_tex_format("dofs(space)", "r");
	convergence_table.set_tex_format("dofs(time)", "r");
	std::cout << std::endl;
	convergence_table.write_text(std::cout);
	
	convergence_table.add_column_to_supercolumn("cycle", "n cells");
	convergence_table.add_column_to_supercolumn("cells", "n cells");
	std::vector<std::string> new_order;
	new_order.emplace_back("n cells");
	new_order.emplace_back("L2");
	convergence_table.set_column_order(new_order);
	if (refine_space && refine_time)
	  convergence_table.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate_log2);

	// compute convergence rates from 3 consecutive errors
	for (unsigned int i = 0; i < L2_error_vals.size(); ++i)
	{
	  if (i < 2)
	  convergence_table.add_value("L2...", std::string("-"));
	  else
	  {
	  double p0 = L2_error_vals[i-2];
	  double p1 = L2_error_vals[i-1];
	  double p2 = L2_error_vals[i];
	  convergence_table.add_value("L2...", std::log(std::fabs((p0-p1) / (p1-p2))) / std::log(2.));
	  }
	}

	std::cout << std::endl;
	convergence_table.write_text(std::cout);
}

template<int dim>
void SpaceTime<dim>::run() {
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

		// create and solve linear system
		setup_system();
		assemble_system();
		solve();
		
		// output results as SVG or VTK files
		output_results(cycle); 

		// Compute the error to the analytical solution
		process_solution(cycle);

		// refine mesh
		if (cycle < max_n_refinement_cycles - 1)
		{
		  space_triangulation.refine_global(refine_space);
		  time_triangulation.refine_global(refine_time);
		}
	}

	print_convergence_table();
}

int main() {
	try {
		deallog.depth_console(2);

		// run the simulation
		SpaceTime<1> space_time_problem(
		  1,     // s ->  spatial FE degree
		  1, //2,     // r -> temporal FE degree
		  0.,    // start_time
		  4.,    // end_time,
		  7,     // max_n_refinement_cycles,
		  true,  // refine_space
		  true   // refine_time
		);
		/*
		SpaceTime<1> space_time_problem(
		  1,     // s ->  spatial FE degree
		  1,     // r -> temporal FE degree
		  0.,    // start_time
		  1.,    // end_time,
		  3,     // max_n_refinement_cycles,
		  true,  // refine_space
		  true   // refine_time
		);
		*/
		/*
		SpaceTime<2> space_time_problem(
		  1,     // s ->  spatial FE degree
		  1,     // r -> temporal FE degree
		  0.,    // start_time
		  1.,    // end_time,
		  3,     // max_n_refinement_cycles,
		  true,  // refine_space
		  true   // refine_time
		);
		*/
		space_time_problem.run();

		// save final grid
		space_time_problem.print_grids("space_grid.svg", "time_grid.svg");
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
