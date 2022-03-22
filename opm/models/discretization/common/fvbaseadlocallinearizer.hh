// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*
  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 2 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.

  Consult the COPYING file in the top-level source directory of this
  module for the precise wording of the license and the list of
  copyright holders.
*/
/*!
 * \file
 *
 * \copydoc Opm::FvBaseAdLocalLinearizer
 */
#ifndef EWOMS_FV_BASE_AD_LOCAL_LINEARIZER_HH
#define EWOMS_FV_BASE_AD_LOCAL_LINEARIZER_HH

#include "fvbaseproperties.hh"

#include <opm/material/densead/Math.hpp>
#include <opm/material/common/Valgrind.hpp>
#include <opm/material/common/Unused.hpp>

#include <dune/istl/bvector.hh>
#include <dune/istl/matrix.hh>

#include <dune/common/fvector.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/timer.hh>

namespace Opm {
// forward declaration
template<class TypeTag>
class FvBaseAdLocalLinearizer;
}

namespace Opm::Properties {

// declare the property tags required for the finite differences local linearizer

namespace TTag {
struct AutoDiffLocalLinearizer {};
} // namespace TTag

// set the properties to be spliced in
template<class TypeTag>
struct LocalLinearizer<TypeTag, TTag::AutoDiffLocalLinearizer>
{ using type = FvBaseAdLocalLinearizer<TypeTag>; };

//! Set the function evaluation w.r.t. the primary variables
template<class TypeTag>
struct Evaluation<TypeTag, TTag::AutoDiffLocalLinearizer>
{
private:
    static const unsigned numEq = getPropValue<TypeTag, Properties::NumEq>();

    using Scalar = GetPropType<TypeTag, Properties::Scalar>;

public:
    using type = DenseAd::Evaluation<Scalar, numEq>;
};

} // namespace Opm::Properties

namespace Opm {

/*!
 * \ingroup FiniteVolumeDiscretizations
 *
 * \brief Calculates the local residual and its Jacobian for a single element of the grid.
 *
 * This class uses automatic differentiation to calculate the partial derivatives (the
 * alternative is finite differences).
 */
template<class TypeTag>
class FvBaseAdLocalLinearizer
{
private:
    using Implementation = GetPropType<TypeTag, Properties::LocalLinearizer>;
    using LocalResidual = GetPropType<TypeTag, Properties::LocalResidual>;
    using Simulator = GetPropType<TypeTag, Properties::Simulator>;
    using Problem = GetPropType<TypeTag, Properties::Problem>;
    using Model = GetPropType<TypeTag, Properties::Model>;
    using PrimaryVariables = GetPropType<TypeTag, Properties::PrimaryVariables>;
    using ElementContext = GetPropType<TypeTag, Properties::ElementContext>;
    using Scalar = GetPropType<TypeTag, Properties::Scalar>;
    using GridView = GetPropType<TypeTag, Properties::GridView>;
    using Element = typename GridView::template Codim<0>::Entity;

    enum { numEq = getPropValue<TypeTag, Properties::NumEq>() };

    using ScalarVectorBlock = Dune::FieldVector<Scalar, numEq>;
    // extract local matrices from jacobian matrix for consistency
    using ScalarMatrixBlock = typename GetPropType<TypeTag, Properties::SparseMatrixAdapter>::MatrixBlock;

    using ScalarLocalBlockVector = Dune::BlockVector<ScalarVectorBlock>;
    using ScalarLocalBlockMatrix = Dune::Matrix<ScalarMatrixBlock>;

public:
    Dune::Timer update_stencil_timer;
    Dune::Timer lin_elem_timer;
    Dune::Timer update_i_timer;
    Dune::Timer update_pv_timer;
    Dune::Timer reset_timer;
    Dune::Timer update_e_timer;
    Dune::Timer update_residual_timer;
    Dune::Timer conv_residual_timer;

    FvBaseAdLocalLinearizer()
        : update_stencil_timer(false),
          lin_elem_timer(false),
          update_i_timer(false),
          update_pv_timer(false),
          reset_timer(false),
          update_e_timer(false),
          update_residual_timer(false),
          conv_residual_timer(false),
          internalElemContext_(0)
    { }

    // copying local linearizer objects around is a very bad idea, so we explicitly
    // prevent it...
    FvBaseAdLocalLinearizer(const FvBaseAdLocalLinearizer&) = delete;

    ~FvBaseAdLocalLinearizer()
    { delete internalElemContext_; }

    double getTimerSum() const
    {
      return update_stencil_timer.elapsed() +
             update_i_timer.elapsed() +
             update_pv_timer.elapsed() +
             reset_timer.elapsed() +
             update_e_timer.elapsed() +
             update_residual_timer.elapsed() +
             conv_residual_timer.elapsed();
    }

    /*!
     * \brief Register all run-time parameters for the local jacobian.
     */
    static void registerParameters()
    { }

    /*!
     * \brief Initialize the local Jacobian object.
     *
     * At this point we can assume that everything has been allocated,
     * although some objects may not yet be completely initialized.
     *
     * \param simulator The simulator object of the simulation.
     */
    void init(Simulator& simulator)
    {
        simulatorPtr_ = &simulator;
        delete internalElemContext_;
        internalElemContext_ = new ElementContext(simulator);
    }

    /*!
     * \brief Compute an element's local Jacobian matrix and evaluate its residual.
     *
     * The local Jacobian for a given context is defined as the derivatives of the
     * residuals of all degrees of freedom featured by the stencil with regard to the
     * primary variables of the stencil's "primary" degrees of freedom. Adding the local
     * Jacobians for all elements in the grid will give the global Jacobian 'grad f(x)'.
     *
     * \param element The grid element for which the local residual and its local
     *                Jacobian should be calculated.
     */
    void linearize(const Element& element)
    {
        linearize(*internalElemContext_, element);
    }

    /*!
     * \brief Compute an element's local Jacobian matrix and evaluate its residual.
     *
     * The local Jacobian for a given context is defined as the derivatives of the
     * residuals of all degrees of freedom featured by the stencil with regard to the
     * primary variables of the stencil's "primary" degrees of freedom. Adding the local
     * Jacobians for all elements in the grid will give the global Jacobian 'grad f(x)'.
     *
     * After calling this method the ElementContext is in an undefined state, so do not
     * use it anymore!
     *
     * \param elemCtx The element execution context for which the local residual and its
     *                local Jacobian should be calculated.
     */
    void linearize(ElementContext& elemCtx, const Element& elem)
    {
        lin_elem_timer.start();
        update_stencil_timer.start();
        elemCtx.updateStencil(elem);
        update_stencil_timer.stop();
        update_i_timer.start();
        elemCtx.updateAllIntensiveQuantities();
        update_i_timer.stop();

        // update the weights of the primary variables for the context
        update_pv_timer.start();
        model_().updatePVWeights(elemCtx);
        update_pv_timer.stop();

        // resize the internal arrays of the linearizer
        reset_timer.start();
        resize_(elemCtx);
        reset_(elemCtx);
        reset_timer.stop();

        // compute the local residual and its Jacobian
        unsigned numPrimaryDof = elemCtx.numPrimaryDof(/*timeIdx=*/0);
        for (unsigned focusDofIdx = 0; focusDofIdx < numPrimaryDof; focusDofIdx++) {
            elemCtx.setFocusDofIndex(focusDofIdx);
            update_e_timer.start();
            elemCtx.updateAllExtensiveQuantities();
            update_e_timer.stop();

            // calculate the local residual
            update_residual_timer.start();
            localResidual_.eval(elemCtx);
            update_residual_timer.stop();

            // convert the local Jacobian matrix and the right hand side from the data
            // structures used by the automatic differentiation code to the conventional
            // ones used by the linear solver.
            conv_residual_timer.start();
            updateLocalLinearization_(elemCtx, focusDofIdx);
            conv_residual_timer.stop();
        }
        lin_elem_timer.stop();
    }

    /*!
     * \brief Return reference to the local residual.
     */
    LocalResidual& localResidual()
    { return localResidual_; }

    /*!
     * \brief Return reference to the local residual.
     */
    const LocalResidual& localResidual() const
    { return localResidual_; }

    /*!
     * \brief Returns the local Jacobian matrix of the residual of a sub-control volume.
     *
     * \param domainScvIdx The local index of the sub control volume to which the primary
     *                     variables are associated with
     * \param rangeScvIdx The local index of the sub control volume which contains the
     *                    local residual
     */
    const ScalarMatrixBlock& jacobian(unsigned domainScvIdx, unsigned rangeScvIdx) const
    { return jacobian_[domainScvIdx][rangeScvIdx]; }

    /*!
     * \brief Returns the local residual of a sub-control volume.
     *
     * \param dofIdx The local index of the sub control volume
     */
    const ScalarVectorBlock& residual(unsigned dofIdx) const
    { return residual_[dofIdx]; }

protected:
    Implementation& asImp_()
    { return *static_cast<Implementation*>(this); }
    const Implementation& asImp_() const
    { return *static_cast<const Implementation*>(this); }

    const Simulator& simulator_() const
    { return *simulatorPtr_; }
    const Problem& problem_() const
    { return simulatorPtr_->problem(); }
    const Model& model_() const
    { return simulatorPtr_->model(); }

    /*!
     * \brief Resize all internal attributes to the size of the
     *        element.
     */
    void resize_(const ElementContext& elemCtx)
    {
        size_t numDof = elemCtx.numDof(/*timeIdx=*/0);
        size_t numPrimaryDof = elemCtx.numPrimaryDof(/*timeIdx=*/0);

        residual_.resize(numDof);
        jacobian_.setSize(numDof, numPrimaryDof);
    }

    /*!
     * \brief Reset the all relevant internal attributes to 0
     */
    void reset_(const ElementContext&)
    {
        residual_ = 0.0;
        jacobian_ = 0.0;
    }

    /*!
     * \brief Updates the current local Jacobian matrix with the partial derivatives of
     *        all equations for the degree of freedom associated with 'focusDofIdx'.
     */
    void updateLocalLinearization_(const ElementContext& elemCtx,
                                   unsigned focusDofIdx)
    {
        const auto& resid = localResidual_.residual();

        for (unsigned eqIdx = 0; eqIdx < numEq; eqIdx++)
            residual_[focusDofIdx][eqIdx] = resid[focusDofIdx][eqIdx].value();

        size_t numDof = elemCtx.numDof(/*timeIdx=*/0);
        for (unsigned dofIdx = 0; dofIdx < numDof; dofIdx++) {
            for (unsigned eqIdx = 0; eqIdx < numEq; eqIdx++) {
                for (unsigned pvIdx = 0; pvIdx < numEq; pvIdx++) {
                    // A[dofIdx][focusDofIdx][eqIdx][pvIdx] is the partial derivative of
                    // the residual function 'eqIdx' for the degree of freedom 'dofIdx'
                    // with regard to the focus variable 'pvIdx' of the degree of freedom
                    // 'focusDofIdx'
                    jacobian_[dofIdx][focusDofIdx][eqIdx][pvIdx] = resid[dofIdx][eqIdx].derivative(pvIdx);
                    Valgrind::CheckDefined(jacobian_[dofIdx][focusDofIdx][eqIdx][pvIdx]);
                }
            }
        }
    }

    Simulator *simulatorPtr_;
    Model *modelPtr_;

    ElementContext *internalElemContext_;

    LocalResidual localResidual_;

    ScalarLocalBlockVector residual_;
    ScalarLocalBlockMatrix jacobian_;
};

} // namespace Opm

#endif
