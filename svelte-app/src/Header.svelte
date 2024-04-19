<script>
    import { Navbar, NavbarBrand, Nav, NavItem, NavLink} from 'sveltestrap';
</script>


<Navbar class="d-flex" color="secondary" light expand="xs">
    <NavbarBrand href="/" style="color: black !important; font-weight: bold !important">제목은 CNN VIS로 하겠습니다. 근데 이제 HuggingFace를 곁들인</NavbarBrand>
    <Nav className="ms-auto" navbar>
      <NavItem>
        <NavLink href="/" style="color: black !important;">VIBA</NavLink>
      </NavItem>
      <NavItem>
        <NavLink href="/" style="color: black !important;">GitHub</NavLink>
      </NavItem>
    </Nav>
  </Navbar>