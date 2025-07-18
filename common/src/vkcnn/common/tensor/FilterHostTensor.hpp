#pragma once

#include "vkcnn/common/tensor/FilterLayout.hpp"
#include "vkcnn/common/tensor/FitlerDescriptor.hpp"
#include "vkcnn/common/tensor/FloatType.hpp"
#include <cassert>
#include <concepts>
#include <cstring>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
namespace vkcnn {

// fwd declarations
class FilterHostTensor;
class FilterHostTensorConstView;

class FilterHostTensorView {
public:
  friend class FilterHostTensorConstView;
  explicit FilterHostTensorView(FilterDescriptor desc, std::byte *buffer)
      : m_desc(desc), m_buffer(buffer) {}

  explicit FilterHostTensorView(FilterHostTensor *tensor);
  FilterHostTensorView(FilterHostTensor &tensor);

  const std::byte *data() const { return m_buffer; }
  std::byte *data() { return m_buffer; }

  std::span<const std::byte> span() const {
    return std::span{m_buffer, m_desc.byteSize()};
  }
  std::span<std::byte> span() { return std::span{m_buffer, m_desc.byteSize()}; }

  const FilterDescriptor &desc() const { return m_desc; }
  const FilterShape &shape() const { return m_desc.shape; }
  FilterLayout layout() const { return m_desc.layout; }
  FloatType type() const { return m_desc.type; }
  std::size_t byteSize() const { return m_desc.byteSize(); }

  fxx_reference at(unsigned int s, unsigned int r, unsigned int c,
                   unsigned int k) {
    std::size_t linearIndex = m_desc.layout(m_desc.shape, s, r, c, k);
    std::size_t offset = linearIndex * m_desc.type.size();
    std::byte *ptr = m_buffer + offset;
    return fxx_reference(reinterpret_cast<void *>(ptr), m_desc.type);
  }

  fxx_const_reference at(unsigned int s, unsigned int r, unsigned int c,
                         unsigned int k) const {
    std::size_t linearIndex = m_desc.layout(m_desc.shape, s, r, c, k);
    std::size_t offset = linearIndex * m_desc.type.size();
    const std::byte *ptr = m_buffer + offset;
    return fxx_const_reference(reinterpret_cast<const void *>(ptr),
                               m_desc.type);
  }

  fxx_reference operator[](unsigned int linearIndex) {
    std::size_t offset = linearIndex * m_desc.type.size();
    std::byte *ptr = m_buffer + offset;
    return fxx_reference(reinterpret_cast<void *>(ptr), m_desc.type);
  }

  fxx_const_reference operator[](unsigned int linearIndex) const {
    std::size_t offset = linearIndex * m_desc.type.size();
    const std::byte *ptr = m_buffer + offset;
    return fxx_const_reference(reinterpret_cast<const void *>(ptr),
                               m_desc.type);
  }

  void assignFrom(const FilterHostTensorConstView &view);

private:
  FilterDescriptor m_desc;
  std::byte *m_buffer;
};

class FilterHostTensorConstView {
public:
  explicit FilterHostTensorConstView(FilterDescriptor desc,
                                     const std::byte *buffer)
      : m_desc(desc), m_buffer(buffer) {}

  FilterHostTensorConstView(FilterHostTensorView view)
      : m_desc(view.m_desc), m_buffer(view.m_buffer) {}

  explicit FilterHostTensorConstView(const FilterHostTensor *tensor);
  FilterHostTensorConstView(const FilterHostTensor &tensor);

  const std::byte *data() const { return m_buffer; }

  std::span<const std::byte> span() const {
    return std::span{m_buffer, m_desc.byteSize()};
  }

  const FilterDescriptor &desc() const { return m_desc; }
  const FilterShape &shape() const { return m_desc.shape; }
  FilterLayout layout() const { return m_desc.layout; }
  FloatType type() const { return m_desc.type; }
  std::size_t byteSize() const { return m_desc.byteSize(); }

  fxx_const_reference at(unsigned int s, unsigned int r, unsigned int c,
                         unsigned int k) const {
    std::size_t linearIndex = m_desc.layout(m_desc.shape, s, r, c, k);
    std::size_t offset = linearIndex * m_desc.type.size();
    const std::byte *ptr = m_buffer + offset;
    return fxx_const_reference(reinterpret_cast<const void *>(ptr),
                               m_desc.type);
  }

  fxx_const_reference operator[](unsigned int linearIndex) const {
    std::size_t offset = linearIndex * m_desc.type.size();
    const std::byte *ptr = m_buffer + offset;
    return fxx_const_reference(reinterpret_cast<const void *>(ptr),
                               m_desc.type);
  }

private:
  FilterDescriptor m_desc;
  const std::byte *m_buffer;
};

class FilterHostTensor {
private:
public:
  template <typename Alloc = std::allocator<std::byte>>
  explicit FilterHostTensor(FilterDescriptor descriptor,
                            std::span<const std::byte> weights,
                            const Alloc &alloc = {})
      : m_desc(descriptor) {
    using allocator_traits = std::allocator_traits<Alloc>;

    Alloc allocator = alloc;
    std::size_t n = m_desc.byteSize();
    std::byte *ptr = allocator_traits::allocate(allocator, n);
    assert(n == weights.size());
    std::memcpy(ptr, weights.data(), n);

    m_storage = std::unique_ptr<std::byte[], std::function<void(std::byte *)>>(
        ptr, [allocator, n](std::byte *p) mutable {
          allocator_traits::deallocate(allocator, p, n);
        });
  }

  template <typename Alloc = std::allocator<std::byte>>
    requires(!std::same_as<Alloc, std::span<std::byte>>)
  explicit FilterHostTensor(FilterDescriptor descriptor,
                            const Alloc &alloc = {})
      : m_desc(descriptor) {
    using allocator_traits = std::allocator_traits<Alloc>;

    Alloc allocator = alloc;
    std::size_t n = m_desc.byteSize();
    std::byte *ptr = allocator_traits::allocate(allocator, n);
    std::memset(ptr, 0, n);
    m_storage = std::unique_ptr<std::byte[], std::function<void(std::byte *)>>(
        ptr, [allocator, n](std::byte *p) mutable {
          allocator_traits::deallocate(allocator, p, n);
        });
  }

  template <typename Alloc = std::allocator<std::byte>>
  explicit FilterHostTensor(FilterDescriptor descriptor,
                            FilterHostTensorConstView view,
                            const Alloc &alloc = {})
      : m_desc(descriptor) {
    using allocator_traits = std::allocator_traits<Alloc>;

    Alloc allocator = alloc;
    std::size_t n = m_desc.byteSize();
    std::byte *ptr = allocator_traits::allocate(allocator, n);
    if (shape() != view.shape()) {
      throw std::runtime_error("Invalid tensor shape! Shapes do not match!");
    }
    if (layout() == view.layout() && type() == view.type()) {
      std::memcpy(ptr, view.data(), n);
    } else {
      FilterHostTensorView{m_desc, ptr}.assignFrom(view);
    }

    m_storage = std::unique_ptr<std::byte[], std::function<void(std::byte *)>>(
        ptr, [allocator, n](std::byte *p) mutable {
          allocator_traits::deallocate(allocator, p, n);
        });
  }

  template <typename Alloc = std::allocator<std::byte>>
  explicit FilterHostTensor(FilterHostTensorConstView view,
                            const Alloc &alloc = {})
      : m_desc(view.desc()) {
    using allocator_traits = std::allocator_traits<Alloc>;

    Alloc allocator = alloc;
    std::size_t n = m_desc.byteSize();
    std::byte *ptr = allocator_traits::allocate(allocator, n);
    std::memcpy(ptr, view.data(), n);

    m_storage = std::unique_ptr<std::byte[], std::function<void(std::byte *)>>(
        ptr, [allocator, n](std::byte *p) mutable {
          allocator_traits::deallocate(allocator, p, n);
        });
  }

  const std::byte *data() const { return m_storage.get(); }
  std::byte *data() { return m_storage.get(); }

  std::span<const std::byte> span() const {
    return std::span{m_storage.get(), m_desc.byteSize()};
  }
  std::span<std::byte> span() {
    return std::span{m_storage.get(), m_desc.byteSize()};
  }

  const FilterDescriptor &desc() const { return m_desc; }
  const FilterShape &shape() const { return m_desc.shape; }
  FilterLayout layout() const { return m_desc.layout; }
  FloatType type() const { return m_desc.type; }
  std::size_t byteSize() const { return m_desc.byteSize(); }

  fxx_reference at(unsigned int s, unsigned int r, unsigned int c,
                   unsigned int k) {
    return FilterHostTensorView{this}.at(s, r, c, k);
  }

  fxx_const_reference at(unsigned int s, unsigned int r, unsigned int c,
                         unsigned int k) const {
    return FilterHostTensorConstView{this}.at(s, r, c, k);
  }

  fxx_reference operator[](unsigned int linearIndex) {
    return FilterHostTensorView{this}[linearIndex];
  }

  fxx_const_reference operator[](unsigned int linearIndex) const {
    return FilterHostTensorConstView{this}[linearIndex];
  }

private:
  FilterDescriptor m_desc;
  std::unique_ptr<std::byte[], std::function<void(std::byte *)>> m_storage;
};

} // namespace vkcnn
